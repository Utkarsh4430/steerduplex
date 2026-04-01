"""Evaluate Moshi on BigBench Audio in speech-to-speech mode."""

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import string
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from tqdm import tqdm

from eval.transcription_utils import transcribe_batch
from inference.generate import MoshiInference

CACHE_DIR = "/fs/gamma-projects/audio/raman/steerd/cache"
DEFAULT_SYSTEM_PROMPT = "Answer the question with just the answer, no explanation."

logger = logging.getLogger(__name__)

load_dotenv() 

def configure_environment() -> None:
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Moshi on BigBench Audio")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to finetuned checkpoint dir")
    parser.add_argument("--hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--device", dest="devices", nargs="+", default=["cuda"])
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_transcription", action="store_true")
    parser.add_argument("--whisper_batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N examples")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def default_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"moshi_results_{timestamp}"


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "output_audios").mkdir(parents=True, exist_ok=True)


def load_bigbench_examples(token: Optional[str] = None) -> List[Dict[str, Any]]:
    logger.info("Loading BigBench Audio raw files from HuggingFace...")
    snapshot_dir = snapshot_download(
        repo_id="ArtificialAnalysis/big_bench_audio",
        repo_type="dataset",
        cache_dir=CACHE_DIR,
        token=token,
        allow_patterns=["metadata.jsonl", "data/*.mp3"],
    )

    metadata_path = Path(snapshot_dir) / "metadata.jsonl"
    examples = []
    with metadata_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            record["audio_path"] = str(Path(snapshot_dir) / record.pop("file_name"))
            examples.append(record)

    logger.info("Loaded %d examples.", len(examples))
    return examples


def load_existing_results(output_json_path: Path) -> List[Dict[str, Any]]:
    if not output_json_path.exists():
        return []
    with output_json_path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {output_json_path}, got {type(data).__name__}")
    return data


def save_results(results: List[Dict[str, Any]], output_json_path: Path) -> None:
    tmp_path = output_json_path.with_suffix(".tmp")
    with tmp_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    os.replace(tmp_path, output_json_path)


def unique_id_for(example: Dict[str, Any]) -> str:
    return f"{example['category']}_{example['id']}"


def normalize_answer(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def is_correct_prediction(prediction: str, gold: str) -> bool:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    if not pred or not target:
        return False
    if pred == target:
        return True
    # Use whole-word boundary matching to avoid false positives on short answers
    # (e.g. gold="a" must not match "answer" or "and")
    return bool(re.search(r"\b" + re.escape(target) + r"\b", pred))


def _compute_accuracy(
    entries: List[Dict[str, Any]], text_field: str, label: str
) -> None:
    scored = [e for e in entries if e.get(text_field) is not None]
    if not scored:
        logger.info("No %s available, skipping accuracy summary.", text_field)
        return

    correct = 0
    by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for entry in scored:
        category = entry["category"]
        by_category[category]["total"] += 1
        matched = is_correct_prediction(entry[text_field], entry["official_answer"])
        if matched:
            correct += 1
            by_category[category]["correct"] += 1
        if text_field == "transcription":
            entry["correct"] = matched

    logger.info(
        "[%s] Overall accuracy: %.1f%% (%d/%d)",
        label,
        100.0 * correct / len(scored),
        correct,
        len(scored),
    )
    for category, stats in sorted(by_category.items()):
        logger.info(
            "[%s]   %s: %.1f%% (%d/%d)",
            label,
            category,
            100.0 * stats["correct"] / stats["total"],
            stats["correct"],
            stats["total"],
        )


def log_accuracy(results: List[Dict[str, Any]], output_json_path: Optional[Path] = None) -> None:
    _compute_accuracy(results, "transcription", "whisper")
    _compute_accuracy(results, "moshi_text", "moshi_text")
    if output_json_path is not None:
        save_results(results, output_json_path)


def _run_inference_example(
    model: MoshiInference,
    output_dir: Path,
    example: Dict[str, Any],
    system_prompt: str,
    max_duration: float,
) -> Optional[Dict[str, Any]]:
    unique_id = unique_id_for(example)

    tmp_path = None  # type: Optional[str]
    try:
        audio_array, sample_rate = librosa.load(example["audio_path"], sr=None, mono=True)
        audio_array = np.asarray(audio_array, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio_array, sample_rate)

        out_audio, out_sr, out_text_tokens = model.generate(
            user_audio_path=tmp_path,
            system_prompt=system_prompt,
            max_duration_sec=max_duration,
        )
    except Exception:
        logger.exception("Inference failed for %s, skipping.", unique_id)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    moshi_text = "".join(t for t in out_text_tokens if not t.startswith("["))

    rel_path = Path("output_audios") / f"{unique_id}.wav"
    abs_path = output_dir / rel_path
    sf.write(abs_path, out_audio, out_sr)

    return {
        "unique_id": unique_id,
        "id": example["id"],
        "category": example["category"],
        "official_answer": example["official_answer"],
        "output_audio_path": rel_path.as_posix(),
        "moshi_text": moshi_text,
        "transcription": None,
        "correct": None,
    }


def _inference_worker(
    device: str,
    examples: List[Dict[str, Any]],
    output_dir: Path,
    system_prompt: str,
    max_duration: float,
    hf_repo: str,
    checkpoint: Optional[str],
    seed: int,
    greedy: bool,
    result_queue: mp.Queue,
) -> None:
    configure_environment()
    setup_logging()

    logger.info("Worker starting on %s with %d examples.", device, len(examples))
    try:
        model = MoshiInference(
            hf_repo_id=hf_repo,
            checkpoint_path=checkpoint,
            device=device,
            seed=seed,
            greedy=greedy,
        )
    except Exception:
        logger.exception("Failed to initialize Moshi on %s.", device)
        result_queue.put(("done", device))
        return

    for example in examples:
        result = _run_inference_example(
            model=model,
            output_dir=output_dir,
            example=example,
            system_prompt=system_prompt,
            max_duration=max_duration,
        )
        if result is not None:
            result_queue.put(("result", result))

    logger.info("Worker finished on %s.", device)
    result_queue.put(("done", device))


def run_inference(
    devices: List[str],
    examples: List[Dict[str, Any]],
    output_dir: Path,
    output_json_path: Path,
    system_prompt: str,
    max_duration: float,
    hf_repo: str,
    checkpoint: Optional[str],
    seed: int,
    greedy: bool,
) -> List[Dict[str, Any]]:
    if not devices:
        raise ValueError("At least one device must be provided.")

    results = load_existing_results(output_json_path)
    done_ids = {entry["unique_id"] for entry in results}
    pending_examples = [example for example in examples if unique_id_for(example) not in done_ids]

    logger.info(
        "Starting or resuming inference with %d completed examples and %d pending examples.",
        len(done_ids),
        len(pending_examples),
    )
    if not pending_examples:
        return results

    if len(devices) == 1:
        device = devices[0]
        logger.info("Running Moshi inference in a single process on %s.", device)
        model = MoshiInference(
            hf_repo_id=hf_repo,
            checkpoint_path=checkpoint,
            device=device,
            seed=seed,
            greedy=greedy,
        )
        for example in tqdm(pending_examples, desc="Moshi inference", unit="question"):
            result = _run_inference_example(
                model=model,
                output_dir=output_dir,
                example=example,
                system_prompt=system_prompt,
                max_duration=max_duration,
            )
            if result is None:
                continue

            results.append(result)
            done_ids.add(result["unique_id"])
            save_results(results, output_json_path)
            tqdm.write(f"[{len(results)}/{len(examples)}] Saved {result['output_audio_path']}")

        return results

    logger.info("Pre-caching Moshi model repo %s before spawning workers.", hf_repo)
    snapshot_download(repo_id=hf_repo, cache_dir=CACHE_DIR)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    workers: List[tuple[str, mp.Process]] = []
    for worker_idx, device in enumerate(devices):
        worker_examples = pending_examples[worker_idx::len(devices)]
        if not worker_examples:
            continue
        process = ctx.Process(
            target=_inference_worker,
            args=(
                device,
                worker_examples,
                output_dir,
                system_prompt,
                max_duration,
                hf_repo,
                checkpoint,
                seed,
                greedy,
                result_queue,
            ),
        )
        process.start()
        workers.append((device, process))
        logger.info("Launched Moshi worker on %s for %d examples.", device, len(worker_examples))

    active_devices = {device for device, _ in workers}
    save_interval = 10
    unsaved_results = 0
    pbar = tqdm(total=len(pending_examples), desc="Moshi inference", unit="question")

    while active_devices:
        try:
            item_type, payload = result_queue.get(timeout=1.0)
            if item_type == "result":
                result = payload
                unique_id = result["unique_id"]
                if unique_id in done_ids:
                    continue
                results.append(result)
                done_ids.add(unique_id)
                unsaved_results += 1
                pbar.update(1)
                tqdm.write(f"[{len(results)}/{len(examples)}] Saved {result['output_audio_path']}")
                if unsaved_results >= save_interval:
                    save_results(results, output_json_path)
                    unsaved_results = 0
            elif item_type == "done":
                active_devices.discard(payload)
        except queue.Empty:
            pass

        for device, process in workers:
            if device in active_devices and process.exitcode not in (None, 0):
                logger.error("Worker on %s exited with code %s.", device, process.exitcode)
                active_devices.discard(device)

    pbar.close()

    for _, process in workers:
        process.join()

    while True:
        try:
            item_type, payload = result_queue.get_nowait()
            if item_type != "result":
                continue
            result = payload
            unique_id = result["unique_id"]
            if unique_id in done_ids:
                continue
            results.append(result)
            done_ids.add(unique_id)
        except queue.Empty:
            break

    save_results(results, output_json_path)
    return results


def run_transcription(
    results: List[Dict[str, Any]],
    output_dir: Path,
    device: str,
    batch_size: int,
    output_json_path: Path,
) -> None:
    pending_entries = [entry for entry in results if entry.get("transcription") is None]
    if not pending_entries:
        logger.info("All outputs already have transcriptions, skipping Whisper.")
        return

    audio_paths = [str(output_dir / entry["output_audio_path"]) for entry in pending_entries]
    logger.info("Transcribing %d output audios with Whisper large v3...", len(audio_paths))
    texts = transcribe_batch(audio_paths, device=device, batch_size=batch_size)

    for entry, text in zip(pending_entries, texts):
        entry["transcription"] = text

    save_results(results, output_json_path)
    logger.info("Saved transcriptions to %s", output_json_path)


def main() -> None:
    configure_environment()
    args = parse_args()
    setup_logging()

    output_dir = Path(args.output_dir or default_output_dir())
    output_json_path = output_dir / "output.json"
    ensure_output_dirs(output_dir)

    examples = load_bigbench_examples(token=os.environ.get("HF_TOKEN"))
    category_counts = Counter(example["category"] for example in examples)
    for category, count in sorted(category_counts.items()):
        logger.info("  %s: %d examples", category, count)

    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        examples = examples[: min(args.limit, len(examples))]
        logger.info("Limiting evaluation to first %d examples.", len(examples))

    existing_results = load_existing_results(output_json_path)
    existing_ids = {entry["unique_id"] for entry in existing_results}
    target_ids = {unique_id_for(example) for example in examples}
    missing_ids = target_ids - existing_ids

    if missing_ids:
        logger.info(
            "Preparing Moshi inference across %d device(s): %s",
            len(args.devices),
            ", ".join(args.devices),
        )

        results = run_inference(
            devices=args.devices,
            examples=examples,
            output_dir=output_dir,
            output_json_path=output_json_path,
            system_prompt=args.system_prompt,
            max_duration=args.max_duration,
            hf_repo=args.hf_repo,
            checkpoint=args.checkpoint,
            seed=args.seed,
            greedy=args.greedy,
        )
    else:
        logger.info("All %d examples already have generated outputs. Skipping Moshi inference.", len(examples))
        results = existing_results

    if not args.skip_transcription:
        run_transcription(
            results=results,
            output_dir=output_dir,
            device=args.devices[0],
            batch_size=args.whisper_batch_size,
            output_json_path=output_json_path,
        )
    else:
        logger.info("Skipping transcription as requested.")

    final_results = load_existing_results(output_json_path)
    log_accuracy(final_results, output_json_path=output_json_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
