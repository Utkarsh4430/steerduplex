"""Phase 2 of VoiceBench evaluation: ASR + LLM judge.

Stage B (ASR): local Parakeet, dispatched to a separate conda env via subprocess
(so the Moshi env doesn't need nemo_toolkit). See asr_parakeet.py.

Stage C (judge): for `open` / `qa` splits only, call gpt-4o-mini through the
LiteLLM proxy and attach the raw score list (n=3) to each record. Programmatic
evaluators (ifeval, bbh, harm) skip the LLM call entirely — scoring happens in
summarize.py using the transcription as the response text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from eval.voicebench.splits import (
    SDQA_DEFAULT_REGIONS,
    SUPPORTED_SUBSETS,
    SplitSpec,
    build_specs,
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_WORKERS = 32
DEFAULT_ASR_ENV_PYTHON = (
    "/mnt/efs/ramaneswaranselvakumar/miniconda3/envs/raman_fdb_v1/bin/python"
)
DEFAULT_ASR_DEVICE = "cuda:0"
DEFAULT_ASR_BATCH_SIZE = 32
JUDGE_SAVE_EVERY = 50

# Prompts below are verbatim ports of VoiceBench/api_judge.py (meta_prompt_open,
# meta_prompt_qa). Do not edit without keeping them in sync — leaderboard
# comparability depends on identical prompts.

META_PROMPT_OPEN = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: {prompt}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don’t need to provide any explanations.
"""

META_PROMPT_QA = """### Question
{prompt}

### Reference answer
{reference}

### Candidate answer
{response}

Is the candidate answer correct based on the question and reference answer?
Please only output a single "Yes" or "No". Do not output anything else."""


# ---------------------------------------------------------------------------
# Setup / IO
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _normalize_litellm_base(base_url: str) -> str:
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b + "/v1"


def _resolve_litellm_config(cli_base: Optional[str]) -> tuple[str, str]:
    base = _normalize_litellm_base(cli_base or os.environ.get("LITELLM_BASE_URL", ""))
    key = os.environ.get("LITELLM_API_KEY", "")
    if not base or base == "/v1":
        raise SystemExit("ERROR: LITELLM_BASE_URL not set (pass --litellm-base or export it).")
    if not key:
        raise SystemExit("ERROR: LITELLM_API_KEY not set.")
    return base, key


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        json.dump(results, handle, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Stage B: Parakeet ASR (subprocess into a separate conda env)
# ---------------------------------------------------------------------------

def run_parakeet_asr(
    output_dir: Path,
    specs: List[SplitSpec],
    asr_env_python: str,
    asr_device: str,
    batch_size: int,
    force: bool = False,
) -> None:
    """Dispatch asr_parakeet.py to a separate conda env.

    Collapses SplitSpec back to the subset/sdqa_regions CLI shape that
    asr_parakeet.py expects.
    """
    top_splits: List[str] = []
    sdqa_regions: List[str] = []
    for spec in specs:
        if spec.subset == "sd-qa":
            if "sd-qa" not in top_splits:
                top_splits.append("sd-qa")
            if spec.hf_split not in sdqa_regions:
                sdqa_regions.append(spec.hf_split)
        elif spec.name not in top_splits:
            top_splits.append(spec.name)

    cmd: List[str] = [
        asr_env_python,
        "-m", "eval.voicebench.asr_parakeet",
        "--output_dir", str(output_dir),
        "--splits", *top_splits,
        "--asr_device", asr_device,
        "--batch_size", str(batch_size),
    ]
    if sdqa_regions:
        cmd.extend(["--sdqa_regions", *sdqa_regions])
    if force:
        cmd.append("--force")

    # The foreign env won't have steerduplex's src/ on sys.path by default;
    # inject it so `-m eval.voicebench.asr_parakeet` resolves.
    src_root = Path(__file__).resolve().parents[2]  # .../steerduplex/src
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{existing_pp}" if existing_pp else str(src_root)
    )

    logger.info("Stage B (ASR) — invoking %s", asr_env_python)
    logger.info("  cmd: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, env=env, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"ERROR: asr_env_python not found: {asr_env_python}. "
            "Pass --asr_env_python to point at a python binary with nemo_toolkit installed."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Parakeet ASR subprocess failed (exit={exc.returncode}).") from exc


# ---------------------------------------------------------------------------
# Stage C: LLM judge (open / qa)
# ---------------------------------------------------------------------------

def _judge_one(client: OpenAI, model: str, evaluator: str,
               entry: Dict[str, Any]) -> List[str]:
    """Return the raw list of n=3 score strings from the judge."""
    response_text = entry.get("transcription") or ""
    prompt = entry.get("prompt", "")
    if evaluator == "open":
        user_msg = META_PROMPT_OPEN.replace("{prompt}", prompt).replace("{response}", response_text)
    elif evaluator == "qa":
        reference = entry.get("reference", "")
        user_msg = (META_PROMPT_QA
                    .replace("{prompt}", prompt)
                    .replace("{reference}", reference)
                    .replace("{response}", response_text))
    else:
        raise ValueError(f"LLM judge not applicable to evaluator={evaluator}")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant who tries to help answer the user's question."},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.5,
        top_p=0.95,
        n=3,
        max_tokens=1024,
    )
    return [choice.message.content.strip() for choice in resp.choices]


def judge_split(
    split_dir: Path,
    evaluator: str,
    model: str,
    litellm_base: str,
    litellm_key: str,
    num_workers: int,
) -> None:
    """Run LLM judging for `open` or `qa` splits. Writes split_dir/llm_judge_output.json."""
    output_json = split_dir / "output.json"
    judged_json = split_dir / "llm_judge_output.json"

    results = load_results(output_json)

    # Resume: restore previously-written scores, but only when the transcription
    # they were computed against is identical to the current one. This prevents
    # Stage B (Parakeet) re-transcription from accidentally resurrecting stale
    # scores that graded a different (e.g. corrupted JSON) response.
    if judged_json.exists():
        prev = {r["unique_id"]: r for r in load_results(judged_json)}
        for r in results:
            p = prev.get(r["unique_id"])
            if p is None or p.get("score") is None or r.get("score") is not None:
                continue
            if p.get("transcription") == r.get("transcription"):
                r["score"] = p["score"]

    pending = [r for r in results if r.get("score") is None]
    if not pending:
        save_results(results, judged_json)
        logger.info("[%s] all scores present; skipping LLM judge.", split_dir.name)
        return

    logger.info("[%s] LLM-judging %d entries (%s, model=%s, workers=%d)",
                split_dir.name, len(pending), evaluator, model, num_workers)

    client = OpenAI(base_url=litellm_base, api_key=litellm_key)

    completed_since_save = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_judge_one, client, model, evaluator, entry): entry
                   for entry in pending}
        with tqdm(total=len(pending), desc=f"{split_dir.name} judge", unit="ex",
                  dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                entry = futures[fut]
                try:
                    entry["score"] = fut.result()
                    entry.pop("judge_error", None)
                except Exception as exc:
                    logger.warning("[%s] judge failed for %s: %s",
                                   split_dir.name, entry.get("unique_id"), exc)
                    entry["judge_error"] = str(exc)
                pbar.update(1)
                completed_since_save += 1
                if completed_since_save >= JUDGE_SAVE_EVERY:
                    save_results(results, judged_json)
                    completed_since_save = 0

    save_results(results, judged_json)
    n_ok = sum(1 for r in results if r.get("score") is not None)
    logger.info("[%s] LLM judge done: %d/%d scored.", split_dir.name, n_ok, len(results))


# ---------------------------------------------------------------------------
# Orchestrator entry
# ---------------------------------------------------------------------------

def run_judge_pipeline(
    output_dir: Path,
    specs: List[SplitSpec],
    litellm_base: Optional[str],
    num_workers: int,
    judge_model: str,
) -> None:
    """Stage C: run (or resume) the LLM judge across all splits.

    For programmatic evaluators (ifeval, bbh, harm), simply mirror output.json
    into llm_judge_output.json so summarize.py has a uniform input.
    """
    need_llm = any(spec.evaluator in ("open", "qa") for spec in specs)
    if need_llm:
        litellm_base, litellm_key = _resolve_litellm_config(litellm_base)
    else:
        litellm_base, litellm_key = "", ""

    for i, spec in enumerate(specs, 1):
        split_dir = output_dir / spec.name
        output_json = split_dir / "output.json"
        if not output_json.exists():
            logger.warning("[%d/%d] Skipping %s (no output.json)", i, len(specs), spec.name)
            continue

        logger.info("[%d/%d] Stage C: %s (evaluator=%s)",
                    i, len(specs), spec.name, spec.evaluator)

        if spec.evaluator in ("open", "qa"):
            judge_split(split_dir, spec.evaluator, judge_model,
                        litellm_base, litellm_key, num_workers)
        else:
            # Programmatic evaluators: copy output.json → llm_judge_output.json for uniformity.
            dst = split_dir / "llm_judge_output.json"
            save_results(load_results(output_json), dst)
            logger.info("[%s] no LLM judge needed for evaluator=%s.", spec.name, spec.evaluator)


# ---------------------------------------------------------------------------
# Standalone CLI (ASR + judge + summary)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceBench Phase 2 (ASR + judge) + Phase 3")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=list(SUPPORTED_SUBSETS))
    parser.add_argument("--sdqa_regions", nargs="+", default=None)
    parser.add_argument("--litellm-base", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--asr_env_python", type=str, default=DEFAULT_ASR_ENV_PYTHON,
                        help="Python binary for the Parakeet ASR subprocess.")
    parser.add_argument("--asr_device", type=str, default=DEFAULT_ASR_DEVICE)
    parser.add_argument("--asr_batch_size", type=int, default=DEFAULT_ASR_BATCH_SIZE)
    parser.add_argument("--force_asr", action="store_true",
                        help="Re-transcribe every record from scratch (wipes scores too).")
    parser.add_argument("--skip_asr", action="store_true")
    parser.add_argument("--skip_judge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    output_dir = Path(args.output_dir)

    sdqa_regions = args.sdqa_regions
    if "sd-qa" in args.splits and sdqa_regions is None:
        logger.warning("sd-qa: defaulting to regions=%s (see --sdqa_regions).",
                       list(SDQA_DEFAULT_REGIONS))
        sdqa_regions = list(SDQA_DEFAULT_REGIONS)
    specs = build_specs(args.splits, sdqa_regions or [])

    if not args.skip_asr:
        run_parakeet_asr(
            output_dir=output_dir,
            specs=specs,
            asr_env_python=args.asr_env_python,
            asr_device=args.asr_device,
            batch_size=args.asr_batch_size,
            force=args.force_asr,
        )

    if not args.skip_judge:
        run_judge_pipeline(
            output_dir=output_dir,
            specs=specs,
            litellm_base=args.litellm_base,
            num_workers=args.workers,
            judge_model=args.judge_model,
        )

    # Always regenerate the summary so the markdown stays in sync with state.
    from eval.voicebench import summarize
    summarize.write_summary(output_dir=output_dir, specs=specs, run_args=vars(args))


if __name__ == "__main__":
    main()
