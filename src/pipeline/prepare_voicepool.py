"""Prepare voice pool for the SteerDuplex pipeline.

Supports two sources:
1. LibriSpeech: has metadata JSON with ref_text, gender, duration
2. Common Voice: raw WAV files only — runs Whisper to get ref_text, probes duration

Multi-GPU: Whisper workers are spawned across GPUs based on available memory.
Multi-node safe: each transcription uses atomic file claiming, so N nodes
can run this concurrently without duplicate work.

Outputs pool.jsonl in the format expected by pipeline.assign_voices.

Usage:
    # LibriSpeech only (existing)
    python -m pipeline.prepare_voicepool

    # Add Common Voice pool (extracts tar, transcribes with Whisper)
    python -m pipeline.prepare_voicepool --add_common_voice data/voicepool/common_voice_audios.tar.gz

    # Common Voice only
    python -m pipeline.prepare_voicepool --skip_libri --add_common_voice data/voicepool/common_voice_audios.tar.gz
"""

import argparse
import json
import logging
import os
import random
import tarfile
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.multiprocessing as mp

logging.getLogger("whisper").setLevel(logging.ERROR)


def load_pool_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LibriSpeech source
# ---------------------------------------------------------------------------
def prepare_librispeech(
    voicepool_dir: Path,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    split: str = "both",
) -> list[dict]:
    splits = []
    if split in ("train", "both"):
        splits.append("train")
    if split in ("val", "both"):
        splits.append("val")

    entries = []
    for s in splits:
        pool_json = voicepool_dir / s / f"libri_{s}_pool.json"
        audio_dir = voicepool_dir / s / "audios"

        if not pool_json.exists():
            print(f"[WARN] {pool_json} not found, skipping {s} split")
            continue

        raw = load_pool_json(pool_json)
        for item in raw:
            duration = item.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                continue

            audio_path = audio_dir / f"{item['id']}.wav"
            if not audio_path.exists():
                continue

            entries.append({
                "id": f"libri_{s}_{item['id']}",
                "ref_path": str(audio_path),
                "ref_text": item.get("ref_text", ""),
                "gender": item.get("gender") or "unknown",
                "accent": item.get("accent") or "american",
                "age_range": item.get("age_range") or "unknown",
                "energy": item.get("energy") or "medium",
                "duration": duration,
                "source": f"librispeech_{s}",
            })

    print(f"LibriSpeech: {len(entries)} entries")
    return entries


# ---------------------------------------------------------------------------
# Common Voice source — multi-GPU parallel Whisper transcription
# ---------------------------------------------------------------------------
def extract_common_voice(tar_path: Path, extract_dir: Path) -> Path:
    """Extract Common Voice tar.gz if not already extracted."""
    audio_dir = extract_dir / "audios"
    if audio_dir.exists() and len(list(audio_dir.glob("*.wav"))) > 100:
        count = len(list(audio_dir.glob("*.wav")))
        print(f"Common Voice: already extracted ({count} files in {audio_dir})")
        return audio_dir

    print(f"Extracting {tar_path} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir, filter="data")

    count = len(list(audio_dir.glob("*.wav")))
    print(f"Extracted {count} WAV files")
    return audio_dir


# Per-file transcription result stored as individual JSON to be distributed-safe
def _transcription_path(cache_dir: Path, wav_name: str) -> Path:
    """Path to cached transcription for a single WAV file."""
    return cache_dir / f"{wav_name}.txt"


def _is_transcribed(cache_dir: Path, wav_name: str) -> bool:
    return _transcription_path(cache_dir, wav_name).exists()


# Shared progress counter
_progress_counter: mp.Value = None


def _whisper_worker(
    worker_id: int,
    total_workers: int,
    gpu_id: int,
    work_items: list[tuple[Path, Path]],  # (wav_path, claim_path)
    cache_dir: Path,
    whisper_model_name: str,
    progress_counter,
):
    """Worker process: transcribe files on assigned GPU."""
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
    print(f"[{tag}] Loaded, starting transcription", flush=True)

    for wav_path, claim_path in my_items:
        out_path = _transcription_path(cache_dir, wav_path.name)
        if is_done(out_path):
            with _progress_counter.get_lock():
                _progress_counter.value += 1
            continue

        if not try_claim(claim_path):
            continue  # another node/worker claimed it

        try:
            result = model.transcribe(str(wav_path), language="en")
            text = result["text"].strip()
            # Write transcription atomically
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.rename(out_path)
            release_claim(claim_path)
        except Exception as e:
            # Write empty so we don't retry forever
            out_path.write_text("", encoding="utf-8")
            release_claim(claim_path)
            print(f"  [{tag}] WARN: {wav_path.name}: {e}", flush=True)

        with _progress_counter.get_lock():
            _progress_counter.value += 1

    del model
    torch.cuda.empty_cache()


def _progress_monitor(total: int, progress_counter):
    from tqdm import tqdm
    pbar = tqdm(total=total, desc="Transcribing", unit="file")
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


def transcribe_parallel(
    audio_paths: list[Path],
    cache_dir: Path,
    whisper_model_name: str = "medium",
    max_workers_per_gpu: int = 8,
):
    """Transcribe audio files using multi-GPU Whisper with distributed claiming."""
    from pipeline.distributed import plan_workers

    cache_dir.mkdir(parents=True, exist_ok=True)
    claims_dir = cache_dir / ".claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only untranscribed files
    work_items = []
    for wav_path in audio_paths:
        if _is_transcribed(cache_dir, wav_path.name):
            continue
        claim_path = claims_dir / f"{wav_path.name}.claim"
        work_items.append((wav_path, claim_path))

    if not work_items:
        print("All files already transcribed.")
        return

    random.shuffle(work_items)

    # Plan GPU workers (Whisper medium ≈ 2GB)
    print(f"{len(work_items)} files to transcribe. Planning GPU workers:")
    gpu_plans = plan_workers(
        mem_per_worker_mb=2000,
        max_workers_per_gpu=max_workers_per_gpu,
        min_free_after_mb=1024,
    )

    if not gpu_plans:
        print("[WARN] No GPUs available — falling back to CPU (slow)")
        # Single-threaded CPU fallback
        from pipeline.distributed import release_claim, try_claim
        import whisper
        model = whisper.load_model(whisper_model_name, device="cpu")
        for wav_path, claim_path in work_items:
            out_path = _transcription_path(cache_dir, wav_path.name)
            if out_path.exists():
                continue
            if not try_claim(claim_path):
                continue
            try:
                result = model.transcribe(str(wav_path), language="en")
                out_path.write_text(result["text"].strip(), encoding="utf-8")
            except Exception:
                out_path.write_text("", encoding="utf-8")
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
                    work_items, cache_dir, whisper_model_name,
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
    print(f"Transcription complete: {done}/{len(work_items)}")


def prepare_common_voice(
    tar_path: Path,
    extract_dir: Path,
    min_duration: float = 3.0,
    max_duration: float = 15.0,
    whisper_model: str = "medium",
    max_workers_per_gpu: int = 8,
) -> list[dict]:
    """Extract, transcribe, and build pool entries from Common Voice."""
    audio_dir = extract_common_voice(tar_path, extract_dir)
    cache_dir = extract_dir / "transcriptions"

    all_wavs = sorted(audio_dir.glob("*.wav"))
    print(f"Common Voice: {len(all_wavs)} WAV files found")

    # Parallel Whisper transcription (multi-GPU, multi-node safe)
    transcribe_parallel(all_wavs, cache_dir, whisper_model, max_workers_per_gpu)

    # Build entries with duration filtering
    entries = []
    skipped_duration = 0
    skipped_empty = 0

    for wav_path in all_wavs:
        try:
            info = sf.info(str(wav_path))
            duration = info.duration
        except Exception:
            continue

        if duration < min_duration or duration > max_duration:
            skipped_duration += 1
            continue

        # Read cached transcription
        trans_path = _transcription_path(cache_dir, wav_path.name)
        if not trans_path.exists():
            skipped_empty += 1
            continue
        ref_text = trans_path.read_text(encoding="utf-8").strip()
        if not ref_text or len(ref_text) < 5:
            skipped_empty += 1
            continue

        short_id = wav_path.stem[:12]
        entries.append({
            "id": f"cv_{short_id}",
            "ref_path": str(wav_path),
            "ref_text": ref_text,
            "gender": "unknown",
            "accent": "unknown",
            "age_range": "unknown",
            "energy": "medium",
            "duration": round(duration, 2),
            "source": "common_voice",
        })

    print(f"Common Voice: {len(entries)} entries "
          f"(skipped {skipped_duration} duration, {skipped_empty} empty transcript)")
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare voice pool")
    parser.add_argument("--voicepool_dir", type=str, default="data/voicepool")
    parser.add_argument("--output", type=str, default="data/voices/user/pool.jsonl")
    parser.add_argument("--min_duration", type=float, default=3.0)
    parser.add_argument("--max_duration", type=float, default=15.0)
    parser.add_argument("--split", type=str, default="both", choices=["train", "val", "both"])
    parser.add_argument("--skip_libri", action="store_true", help="Skip LibriSpeech entries")
    parser.add_argument("--add_common_voice", type=str, default=None,
                        help="Path to common_voice_audios.tar.gz")
    parser.add_argument("--cv_extract_dir", type=str, default=None,
                        help="Where to extract Common Voice (default: voicepool_dir/common_voice)")
    parser.add_argument("--whisper_model", type=str, default="medium")
    parser.add_argument("--max_workers_per_gpu", type=int, default=8)
    args = parser.parse_args()

    voicepool_dir = Path(args.voicepool_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_entries = []

    # LibriSpeech
    if not args.skip_libri:
        libri = prepare_librispeech(voicepool_dir, args.min_duration, args.max_duration, args.split)
        all_entries.extend(libri)

    # Common Voice
    if args.add_common_voice:
        tar_path = Path(args.add_common_voice)
        extract_dir = Path(args.cv_extract_dir) if args.cv_extract_dir else voicepool_dir / "common_voice"
        cv = prepare_common_voice(
            tar_path, extract_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            whisper_model=args.whisper_model,
            max_workers_per_gpu=args.max_workers_per_gpu,
        )
        all_entries.extend(cv)

    # Deduplicate by id
    seen = set()
    unique = []
    for e in all_entries:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique.append(e)

    # Write
    with open(output_path, "w") as f:
        for entry in unique:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(unique)} total voice entries to {output_path}")
    sources = {}
    for e in unique:
        src = e["source"]
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
