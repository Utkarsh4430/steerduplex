"""Evaluate Moshi on BigBench Audio in speech-to-speech mode."""

import argparse
import json
import logging
from dotenv import load_dotenv
import os
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
    parser.add_argument("--device", type=str, default="cuda")
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


def run_inference(
    model: MoshiInference,
    examples: List[Dict[str, Any]],
    output_dir: Path,
    output_json_path: Path,
    system_prompt: str,
    max_duration: float,
) -> List[Dict[str, Any]]:
    results = load_existing_results(output_json_path)
    done_ids = {entry["unique_id"] for entry in results}

    logger.info("Starting or resuming inference with %d completed examples.", len(done_ids))
    for example in tqdm(examples, desc="Moshi inference", unit="question"):
        unique_id = unique_id_for(example)
        if unique_id in done_ids:
            continue

        tmp_path = None  # type: Optional[str]
        try:
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
                continue
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        moshi_text = "".join(t for t in out_text_tokens if not t.startswith("["))

        rel_path = Path("output_audios") / f"{unique_id}.wav"
        abs_path = output_dir / rel_path
        sf.write(abs_path, out_audio, out_sr)

        results.append(
            {
                "unique_id": unique_id,
                "id": example["id"],
                "category": example["category"],
                "official_answer": example["official_answer"],
                "output_audio_path": rel_path.as_posix(),
                "moshi_text": moshi_text,
                "transcription": None,
                "correct": None,
            }
        )
        done_ids.add(unique_id)
        save_results(results, output_json_path)
        tqdm.write(f"[{len(results)}/{len(examples)}] Saved {rel_path.as_posix()}")

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
        logger.info("Loading Moshi model (this may take around 60s)...")
        model = MoshiInference(
            hf_repo_id=args.hf_repo,
            checkpoint_path=args.checkpoint,
            device=args.device,
            seed=args.seed,
            greedy=args.greedy,
        )
        logger.info("Model ready. Starting inference on %d examples.", len(examples))

        results = run_inference(
            model=model,
            examples=examples,
            output_dir=output_dir,
            output_json_path=output_json_path,
            system_prompt=args.system_prompt,
            max_duration=args.max_duration,
        )
    else:
        logger.info("All %d examples already have generated outputs. Skipping Moshi inference.", len(examples))
        results = existing_results

    if not args.skip_transcription:
        run_transcription(
            results=results,
            output_dir=output_dir,
            device=args.device,
            batch_size=args.whisper_batch_size,
            output_json_path=output_json_path,
        )
    else:
        logger.info("Skipping transcription as requested.")

    final_results = load_existing_results(output_json_path)
    log_accuracy(final_results, output_json_path=output_json_path)


if __name__ == "__main__":
    main()
