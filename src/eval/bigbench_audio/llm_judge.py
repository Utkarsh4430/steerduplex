"""LLM-as-judge evaluation for BigBench Audio Moshi outputs.

Reads generated outputs from an output.json file and uses an OpenAI model
to judge each transcription as CORRECT or INCORRECT against the official answer.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv("/fs/gamma-projects/audio/raman/steerd/steerduplex/src/eval/fdb_v2/.env")

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-5.4-mini"

JUDGE_PROMPT = """\
Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.
If the CANDIDATE ANSWER contradicts itself, assess the first proposed answer.
If the CANDIDATE ANSWER provides a final answer and working, assess the final answer only.
If the CANDIDATE ANSWER includes irrelevant information, assess only the relevant information.
If the CANDIDATE ANSWER includes a numeric value it is ok if it is spelled e.g. 7 or seven
It is ok if the CANDIDATE ANSWER involves a misspelling of a person's name e.g. Leda or Lida, Autry or Audrie.

The question, for reference only: START QUESTION {question} \\n\\nEND QUESTION

The OFFICIAL ANSWER:{official_answer}

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply only with CORRECT or INCORRECT."""


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge on BigBench Audio Moshi outputs."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to output.json from Moshi eval. Defaults to the latest moshi_results_* dir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write judged output JSON. Defaults to <input_dir>/llm_judge_output.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=JUDGE_MODEL,
        help=f"OpenAI model to use for judging (default: {JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, mp.cpu_count()),
        help="Number of parallel worker processes (default: min(16, cpu_count)).",
    )
    parser.add_argument(
        "--candidate-field",
        type=str,
        default="transcription",
        choices=["transcription", "moshi_text"],
        help="Which field to use as the candidate answer (default: transcription).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only judge the first N examples.",
    )
    return parser.parse_args()


def find_default_input() -> Path:
    """Find the most recently modified moshi_results_* output.json."""
    src_dir = Path(__file__).parent.parent.parent
    candidates = sorted(
        src_dir.glob("moshi_results_*/output.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No moshi_results_*/output.json found. Please pass --input explicitly."
        )
    return candidates[0]


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)


def judge_single(
    entry: Dict[str, Any],
    candidate_field: str,
    model: str,
    client: OpenAI,
) -> str:
    """Call the LLM judge for one entry. Returns 'CORRECT', 'INCORRECT', or 'ERROR'."""
    candidate_answer = entry.get(candidate_field) or ""
    official_answer = entry.get("official_answer", "")
    question = entry.get("question", "")

    prompt = JUDGE_PROMPT.format(
        question=question,
        official_answer=official_answer,
        candidate_answer=candidate_answer,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = response.choices[0].message.content.strip().upper()
        if verdict not in ("CORRECT", "INCORRECT"):
            logger.warning(
                "Unexpected verdict %r for %s, treating as ERROR.", verdict, entry.get("unique_id")
            )
            return "ERROR"
        return verdict
    except Exception:
        logger.exception("OpenAI call failed for %s.", entry.get("unique_id"))
        return "ERROR"


def _worker(
    entries: List[Dict[str, Any]],
    candidate_field: str,
    model: str,
    result_queue: mp.Queue,
) -> None:
    """Worker process: judges each entry and puts results onto the queue."""
    client = OpenAI(base_url="https://us.api.openai.com/v1")
    for entry in entries:
        verdict = judge_single(entry, candidate_field, model, client)
        result_queue.put({"unique_id": entry["unique_id"], "verdict": verdict})
    result_queue.put(None)  # sentinel


def log_accuracy(results: List[Dict[str, Any]]) -> None:
    judged = [r for r in results if r.get("llm_judge") in ("CORRECT", "INCORRECT")]
    if not judged:
        logger.info("No judged results to summarise.")
        return

    by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    correct_total = 0
    for r in judged:
        cat = r.get("category", "unknown")
        by_category[cat]["total"] += 1
        if r["llm_judge"] == "CORRECT":
            correct_total += 1
            by_category[cat]["correct"] += 1

    logger.info(
        "[llm_judge] Overall accuracy: %.1f%% (%d/%d)",
        100.0 * correct_total / len(judged),
        correct_total,
        len(judged),
    )
    for cat, stats in sorted(by_category.items()):
        logger.info(
            "[llm_judge]   %s: %.1f%% (%d/%d)",
            cat,
            100.0 * stats["correct"] / stats["total"],
            stats["correct"],
            stats["total"],
        )


def run_judge(
    entries: List[Dict[str, Any]],
    candidate_field: str,
    model: str,
    num_workers: int,
) -> Dict[str, str]:
    """Run parallel LLM judging. Returns mapping of unique_id -> verdict."""
    if not entries:
        return {}

    ctx = mp.get_context("fork")
    result_queue: mp.Queue = ctx.Queue()

    # Distribute entries across workers
    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(num_workers)]
    for i, entry in enumerate(entries):
        chunks[i % num_workers].append(entry)

    processes = []
    for chunk in chunks:
        if not chunk:
            continue
        p = ctx.Process(
            target=_worker,
            args=(chunk, candidate_field, model, result_queue),
        )
        p.start()
        processes.append(p)

    active = len(processes)
    verdicts: Dict[str, str] = {}

    with tqdm(total=len(entries), desc="LLM judge", unit="example", dynamic_ncols=True) as pbar:
        while active > 0 or not result_queue.empty():
            try:
                item = result_queue.get(timeout=1.0)
            except queue.Empty:
                # Check if any process died unexpectedly
                for p in processes:
                    if p.exitcode not in (None, 0):
                        logger.error("Worker %s exited with code %s.", p.pid, p.exitcode)
                        active -= 1
                continue

            if item is None:
                active -= 1
                continue

            unique_id = item["unique_id"]
            verdict = item["verdict"]
            verdicts[unique_id] = verdict
            pbar.update(1)
            pbar.set_postfix({"last": f"{unique_id}={verdict}"})

    for p in processes:
        p.join()

    return verdicts


def main() -> None:
    setup_logging()
    args = parse_args()

    input_path = Path(args.input) if args.input else find_default_input()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output) if args.output else input_path.parent / "llm_judge_output.json"
    )

    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("Model:  %s", args.model)
    logger.info("Workers: %d", args.workers)
    logger.info("Candidate field: %s", args.candidate_field)

    results = load_results(input_path)
    if args.limit is not None:
        results = results[: args.limit]
        logger.info("Limited to first %d examples.", len(results))

    # Load any previously judged results to allow resuming
    existing_judged: Dict[str, str] = {}
    if output_path.exists():
        try:
            existing_results = load_results(output_path)
            existing_judged = {
                r["unique_id"]: r["llm_judge"]
                for r in existing_results
                if r.get("llm_judge") in ("CORRECT", "INCORRECT", "ERROR")
            }
            logger.info("Resuming: %d already judged.", len(existing_judged))
        except Exception:
            logger.warning("Could not load existing output; starting fresh.")

    pending = [r for r in results if r["unique_id"] not in existing_judged]
    logger.info("%d examples to judge.", len(pending))

    if pending:
        new_verdicts = run_judge(
            entries=pending,
            candidate_field=args.candidate_field,
            model=args.model,
            num_workers=args.workers,
        )
        existing_judged.update(new_verdicts)

    # Merge verdicts back into results list
    for r in results:
        uid = r["unique_id"]
        if uid in existing_judged:
            r["llm_judge"] = existing_judged[uid]
            r["llm_judge_correct"] = existing_judged[uid] == "CORRECT"

    save_results(results, output_path)
    logger.info("Saved judged results to %s", output_path)

    log_accuracy(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Done at %s.", timestamp)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
