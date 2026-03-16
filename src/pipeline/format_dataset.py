"""Phase 5: Format data for moshi-finetune.

Creates:
1. manifest_train.jsonl + manifest_eval.jsonl (with configurable split)
2. Runs Whisper annotation for word-level alignments (multi-GPU, distributed-safe)
3. Injects <system> tagged text into alignments during prompt region

Usage:
    python -m pipeline.format_dataset --config configs/generation.yaml
    python -m pipeline.format_dataset --config configs/generation.yaml --skip_whisper
"""

import argparse
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from pipeline.utils import ensure_dir, get_audio_duration, load_json, load_yaml, save_json

logging.getLogger("whisper").setLevel(logging.ERROR)


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def build_alignments_with_system_prompt(
    metadata: dict,
    whisper_alignments: list | None = None,
) -> list:
    alignments = []
    system_prompt = metadata.get("system_prompt", "")
    if system_prompt:
        prompt_text = wrap_with_system_tags(system_prompt)
        prompt_end = metadata.get("prompt_end_sec", 0.0)
        words = prompt_text.split()
        if words and prompt_end > 0:
            duration_per_word = prompt_end / len(words)
            for i, word in enumerate(words):
                start = round(i * duration_per_word, 3)
                end = round((i + 1) * duration_per_word, 3)
                alignments.append([word, [start, end], "SPEAKER_MAIN"])

    if whisper_alignments:
        alignments.extend(whisper_alignments)

    return alignments


def format_single(wav_path: Path, meta_path: Path, output_dir: Path) -> dict | None:
    metadata = load_json(meta_path)
    out_wav = output_dir / "audio" / wav_path.name
    ensure_dir(out_wav.parent)
    if not out_wav.exists():
        shutil.copy2(wav_path, out_wav)

    # Check for whisper annotation
    whisper_json = out_wav.with_suffix(".json")
    whisper_alignments = None
    if whisper_json.exists():
        whisper_data = load_json(whisper_json)
        whisper_alignments = whisper_data.get("alignments", [])

    alignments = build_alignments_with_system_prompt(metadata, whisper_alignments)

    transcript_data = {
        "alignments": alignments,
        "text_conditions": {
            "prompt_end_sec": str(metadata.get("prompt_end_sec", 0.0)),
            "system_prompt": metadata.get("system_prompt", ""),
        },
        "_metadata": {
            "category": metadata.get("category", ""),
            "data_type": metadata.get("data_type", "standard"),
            "assistant_voice_id": metadata.get("assistant_voice_id", ""),
        },
    }
    save_json(transcript_data, out_wav.with_suffix(".json"))

    return {
        "path": f"audio/{wav_path.name}",
        "duration": metadata["duration_sec"],
    }


# ---------------------------------------------------------------------------
# Multi-GPU Whisper annotation (distributed-safe)
# ---------------------------------------------------------------------------
_progress_counter: mp.Value = None


def _whisper_worker(
    worker_id: int,
    total_workers: int,
    gpu_id: int,
    work_items: list[tuple[Path, Path]],  # (wav_path, claim_path)
    whisper_model_name: str,
    progress_counter,
):
    """Worker: transcribe WAVs on assigned GPU with distributed claiming."""
    from pipeline.distributed import is_done, release_claim, try_claim

    global _progress_counter
    _progress_counter = progress_counter

    device = f"cuda:{gpu_id}"
    my_items = work_items[worker_id::total_workers]
    if not my_items:
        return

    tag = f"W{worker_id}/GPU{gpu_id}"
    print(f"[{tag}] Loading Whisper on {device}, {len(my_items)} files", flush=True)

    import whisper
    model = whisper.load_model(whisper_model_name, device=device)
    print(f"[{tag}] Loaded, starting annotation", flush=True)

    for wav_path, claim_path in my_items:
        json_path = wav_path.with_suffix(".json")
        if is_done(json_path):
            with _progress_counter.get_lock():
                _progress_counter.value += 1
            continue

        if not try_claim(claim_path):
            continue

        try:
            result = model.transcribe(str(wav_path), language="en", word_timestamps=True)
            alignments = []
            for seg in result.get("segments", []):
                for word_info in seg.get("words", []):
                    word = word_info.get("word", "").strip()
                    start = round(word_info.get("start", 0), 3)
                    end = round(word_info.get("end", 0), 3)
                    if word:
                        alignments.append([word, [start, end], "SPEAKER_MAIN"])

            annotation = {"alignments": alignments}
            save_json(annotation, json_path)
            release_claim(claim_path)
        except Exception as e:
            # Write empty annotation so we don't block
            save_json({"alignments": []}, json_path)
            release_claim(claim_path)
            print(f"  [{tag}] WARN: {wav_path.name}: {e}", flush=True)

        with _progress_counter.get_lock():
            _progress_counter.value += 1

    del model
    torch.cuda.empty_cache()


def _progress_monitor(total: int, progress_counter):
    pbar = tqdm(total=total, desc="Whisper annotation", unit="file")
    last = 0
    while True:
        with progress_counter.get_lock():
            done = progress_counter.value
        if done > last:
            pbar.update(done - last)
            last = done
        if done >= total:
            break
        time.sleep(0.5)
    pbar.close()


def run_whisper_parallel(audio_dir: Path, whisper_model: str = "medium"):
    """Run Whisper annotation on all un-annotated WAVs, multi-GPU with claiming."""
    from pipeline.distributed import plan_workers

    # Collect unannotated files
    claims_dir = ensure_dir(audio_dir.parent / ".claims_whisper")
    work_items = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        json_path = wav_path.with_suffix(".json")
        if json_path.exists():
            continue
        claim_path = claims_dir / f"{wav_path.stem}.claim"
        work_items.append((wav_path, claim_path))

    if not work_items:
        print("All files already annotated.")
        return

    random.shuffle(work_items)

    # Plan GPU workers (Whisper medium ~2.5GB per worker)
    print(f"\n{len(work_items)} files need Whisper annotation. Planning workers:")
    gpu_plans = plan_workers(
        mem_per_worker_mb=2500,
        max_workers_per_gpu=8,
        min_free_after_mb=2048,
    )

    if not gpu_plans:
        # CPU fallback — single process
        print("[WARN] No GPUs available for Whisper. Running on CPU (slow).")
        from pipeline.distributed import release_claim, try_claim
        import whisper
        model = whisper.load_model(whisper_model, device="cpu")
        for wav_path, claim_path in tqdm(work_items, desc="Whisper (CPU)"):
            json_path = wav_path.with_suffix(".json")
            if json_path.exists():
                continue
            if not try_claim(claim_path):
                continue
            try:
                result = model.transcribe(str(wav_path), language="en", word_timestamps=True)
                alignments = []
                for seg in result.get("segments", []):
                    for w in seg.get("words", []):
                        word = w.get("word", "").strip()
                        if word:
                            alignments.append([word, [round(w["start"], 3), round(w["end"], 3)], "SPEAKER_MAIN"])
                save_json({"alignments": alignments}, json_path)
            except Exception:
                save_json({"alignments": []}, json_path)
            release_claim(claim_path)
        return

    total_workers = min(sum(p.num_workers for p in gpu_plans), len(work_items))
    print(f"Launching {total_workers} Whisper workers across {len(gpu_plans)} GPUs\n")

    mp.set_start_method("spawn", force=True)
    progress_counter = mp.Value("i", 0)

    import threading
    monitor = threading.Thread(
        target=_progress_monitor,
        args=(len(work_items), progress_counter),
        daemon=True,
    )
    monitor.start()

    processes = []
    global_worker_id = 0
    for plan in gpu_plans:
        for _ in range(plan.num_workers):
            p = mp.Process(
                target=_whisper_worker,
                args=(
                    global_worker_id, total_workers, plan.gpu_id,
                    work_items, whisper_model,
                    progress_counter,
                ),
            )
            p.start()
            processes.append(p)
            global_worker_id += 1

    for p in processes:
        p.join()

    monitor.join(timeout=5)

    with progress_counter.get_lock():
        done = progress_counter.value
    print(f"Whisper annotation complete: {done}/{len(work_items)}")


def main():
    parser = argparse.ArgumentParser(description="Format dataset for moshi-finetune")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--assembled_dir", type=str, default=None)
    parser.add_argument("--skip_whisper", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    full_cfg = load_yaml(args.config)
    cfg = full_cfg["dataset"]
    quality_cfg = full_cfg["quality"]

    assembled_dir = Path(args.assembled_dir or full_cfg["assembly"]["output_dir"])
    output_dir = ensure_dir(cfg["output_dir"])
    audio_dir = ensure_dir(output_dir / "audio")

    from pipeline.distributed import is_done, release_claim, try_claim

    wav_files = sorted(assembled_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    # Step 1: Copy WAVs (distributed-safe via claiming)
    print("=== Step 1: Copying audio ===")
    claims_dir = ensure_dir(output_dir / ".claims_fmt")
    copied = 0
    for wav_path in tqdm(wav_files, desc="Copying"):
        dest = audio_dir / wav_path.name
        if is_done(dest):
            continue
        claim_path = claims_dir / f"{wav_path.stem}.claim"
        if not try_claim(claim_path):
            continue
        shutil.copy2(wav_path, dest)
        release_claim(claim_path)
        copied += 1
    print(f"  Copied {copied} new files")

    # Step 2: Whisper annotation (multi-GPU, distributed-safe)
    if not args.skip_whisper:
        print("\n=== Step 2: Whisper annotation ===")
        run_whisper_parallel(audio_dir, quality_cfg.get("whisper_model", "medium"))

    # Step 3: Build final dataset with train/eval split
    print("\n=== Step 3: Building dataset ===")
    entries = []
    for wav_path in tqdm(wav_files, desc="Formatting"):
        meta_path = assembled_dir / f"{wav_path.stem}_meta.json"
        if not meta_path.exists():
            continue
        entry = format_single(wav_path, meta_path, output_dir)
        if entry:
            entries.append(entry)

    # Split into train/eval
    eval_ratio = cfg.get("eval_split_ratio", 0.05)
    random.shuffle(entries)
    n_eval = max(1, int(len(entries) * eval_ratio))
    eval_entries = entries[:n_eval]
    train_entries = entries[n_eval:]

    # Write manifests
    train_manifest = output_dir / "manifest_train.jsonl"
    eval_manifest = output_dir / "manifest_eval.jsonl"
    for path, data in [(train_manifest, train_entries), (eval_manifest, eval_entries)]:
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    # Also write combined manifest for compatibility
    combined = output_dir / "manifest.jsonl"
    with open(combined, "w") as f:
        for entry in train_entries + eval_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDataset ready:")
    print(f"  Train: {len(train_entries)} conversations → {train_manifest}")
    print(f"  Eval:  {len(eval_entries)} conversations → {eval_manifest}")
    print(f"  Audio: {audio_dir}")


if __name__ == "__main__":
    main()
