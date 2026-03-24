#!/usr/bin/env python3
"""
Batch annotate audio files with WhisperX — 4-GPU distributed version.

Uses multiprocessing with a shared Queue: each worker owns one GPU and pulls
jobs from the queue, giving natural load balancing across variable-length files.

Output format identical to single-GPU version:
  {"alignments": [["word", [start, end], "SPEAKER_MAIN"], ...]}

Restore: already-done (.json) and skipped-error (.json.err) files are filtered
before work is distributed, so re-runs pick up exactly where they left off.
"""

import argparse
import gc
import gzip
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import subprocess

import numpy as np
import torch
import whisperx
from tqdm import tqdm

SAMPLE_RATE = 16_000
_SENTINEL = None  # signals worker to stop


def load_audio_channel0(path: str) -> np.ndarray:
    """Load only channel 0 of audio via ffmpeg, resampled to SAMPLE_RATE.
    Uses -map_channel 0.0.0 to select channel 0 instead of mixing down."""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", path,
        "-af", "pan=mono|c0=c0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def init_logging(verbose: bool = False, prefix: str = ""):
    fmt = f"[%(asctime)s][{prefix}%(name)s][%(levelname)s] - %(message)s"
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_and_rename(path: Path, content: str):
    tmp = str(path) + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(content)
    os.rename(tmp, path)


def load_audio_paths(egs_path: Path) -> list[Path]:
    open_fn = gzip.open if str(egs_path).lower().endswith(".gz") else open
    with open_fn(egs_path, "rb") as fp:
        lines = fp.readlines()
    paths = []
    for line in tqdm(lines, desc="Loading egs", file=sys.stderr):
        d = json.loads(line)
        paths.append(Path(d["path"]))
    return paths


def load_audio_paths_from_dir(directory: Path) -> list[Path]:
    paths = sorted(directory.rglob("*.wav"))
    return paths


# ---------------------------------------------------------------------------
# Core processing (runs inside worker process)
# ---------------------------------------------------------------------------

def process_one(
    audio_path: Path,
    out_file: Path,
    model,
    align_model,
    align_metadata,
    language: str,
    device: str,
    batch_size: int,
):
    audio = load_audio_channel0(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    alignments = []
    for segment in result["segments"]:
        if "words" not in segment:
            logger = logging.getLogger(__name__)
            logger.warning("No words in segment for %s: %r", audio_path, segment)
            continue
        for word in segment["words"]:
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                continue
            alignments.append([word["word"], [start, end], "SPEAKER_MAIN"])

    output = {"alignments": alignments}
    write_and_rename(out_file, json.dumps(output, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker_fn(
    gpu_id: int,
    queue: mp.Queue,
    counter: mp.Value,
    error_counter: mp.Value,
    whisper_model_name: str,
    lang: str,
    batch_size: int,
    verbose: bool,
):
    """Runs in a child process. Owns GPU `gpu_id`, drains `queue`."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    compute_type = "float16"

    init_logging(verbose, prefix=f"GPU{gpu_id}|")
    logger = logging.getLogger(__name__)

    logger.info("GPU %d — loading WhisperX model '%s'", gpu_id, whisper_model_name)
    model = whisperx.load_model(
        whisper_model_name, device, compute_type=compute_type, language=lang
    )

    logger.info("GPU %d — loading alignment model for '%s'", gpu_id, lang)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=lang, device=device
    )
    logger.info("GPU %d — ready, waiting for work", gpu_id)

    local_done = 0
    local_errors = 0
    local_start = time.time()

    while True:
        item = queue.get()
        if item is _SENTINEL:
            break

        path = Path(item)
        out_file = path.with_suffix(".json")
        err_file = path.with_suffix(".json.err")

        t0 = time.time()
        try:
            if path.stat().st_size < 1000:
                logger.warning("GPU %d — small file, skipping: %s", gpu_id, path)
                with counter.get_lock():
                    counter.value += 1
                continue

            process_one(
                path,
                out_file,
                model=model,
                align_model=align_model,
                align_metadata=align_metadata,
                language=lang,
                device=device,
                batch_size=batch_size,
            )
            elapsed = time.time() - t0
            local_done += 1
            with counter.get_lock():
                counter.value += 1

            logger.info(
                "GPU %d — done [local %d] %.1fs — %s",
                gpu_id, local_done, elapsed, path.name,
            )

        except Exception as err:
            if "cuda" in repr(err).lower():
                logger.error("GPU %d — CUDA error, re-raising: %s", gpu_id, err)
                raise
            logger.exception("GPU %d — error processing %s", gpu_id, path)
            err_file.touch()
            local_errors += 1
            with error_counter.get_lock():
                error_counter.value += 1
            gc.collect()
            torch.cuda.empty_cache()

    wall = time.time() - local_start
    logger.info(
        "GPU %d — finished. local_done=%d local_errors=%d wall=%.0fs",
        gpu_id, local_done, local_errors, wall,
    )


# ---------------------------------------------------------------------------
# ETA monitor (runs as a thread in main process)
# ---------------------------------------------------------------------------

def eta_monitor(counter: mp.Value, error_counter: mp.Value, total: int, stop_event):
    """Prints overall progress + ETA every 30 seconds."""
    logger = logging.getLogger("eta_monitor")
    start = time.time()
    prev_done = 0

    while not stop_event.is_set():
        stop_event.wait(timeout=30)

        done = counter.value
        errors = error_counter.value
        elapsed = time.time() - start
        remaining = total - done

        rate_overall = done / elapsed if elapsed > 0 else 0
        # windowed rate between last two samples
        interval_done = done - prev_done
        rate_window = interval_done / 30 if interval_done > 0 else rate_overall
        prev_done = done

        if rate_overall > 0 and remaining > 0:
            eta_s = remaining / rate_overall
            eta_str = _fmt_duration(eta_s)
        else:
            eta_str = "?"

        pct = 100 * done / total if total > 0 else 0
        logger.info(
            "PROGRESS  %d/%d (%.1f%%)  errors=%d  "
            "rate=%.2f f/s (window) / %.2f f/s (overall)  "
            "elapsed=%s  ETA=%s",
            done, total, pct, errors,
            rate_window, rate_overall,
            _fmt_duration(elapsed), eta_str,
        )


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

@dataclass
class Params:
    egs: Path
    verbose: bool
    lang: str
    whisper_model: str
    batch_size: int
    rerun_errors: bool
    num_gpus: int


def run(params: Params):
    import threading

    init_logging(params.verbose, prefix="MAIN|")
    logger = logging.getLogger(__name__)

    logger.info("Loading audio paths from: %s", params.egs)
    if params.egs.is_dir():
        paths = load_audio_paths_from_dir(params.egs)
    else:
        paths = load_audio_paths(params.egs)
    logger.info("Total files in egs: %d", len(paths))

    # ---- Restore: filter already-done and skipped-error files ----
    logger.info("Checking existing outputs (restore)...")
    todo = []
    already_done = 0
    already_errored = 0
    for p in tqdm(paths, desc="Checking existing", file=sys.stderr):
        out_file = p.with_suffix(".json")
        err_file = p.with_suffix(".json.err")
        if out_file.exists():
            already_done += 1
            continue
        if err_file.exists() and not params.rerun_errors:
            already_errored += 1
            continue
        todo.append(p)

    total = len(todo)
    logger.info(
        "Restore complete: %d already done, %d previously errored (skipped), %d to process",
        already_done, already_errored, total,
    )

    if total == 0:
        logger.info("Nothing to do — all files already processed.")
        return

    # ---- Build shared queue ----
    queue: mp.Queue = mp.Queue()
    for p in todo:
        queue.put(str(p))
    # Add one sentinel per worker
    for _ in range(params.num_gpus):
        queue.put(_SENTINEL)

    # ---- Shared counters ----
    counter = mp.Value("i", 0)        # completed
    error_counter = mp.Value("i", 0)  # errors

    # ---- Start ETA monitor thread ----
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=eta_monitor,
        args=(counter, error_counter, total, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    # ---- Launch worker processes ----
    logger.info("Launching %d GPU workers...", params.num_gpus)
    workers = []
    for gpu_id in range(params.num_gpus):
        p = mp.Process(
            target=worker_fn,
            args=(
                gpu_id,
                queue,
                counter,
                error_counter,
                params.whisper_model,
                params.lang,
                params.batch_size,
                params.verbose,
            ),
            daemon=False,
        )
        p.start()
        workers.append(p)
        logger.info("Worker for GPU %d started (pid=%d)", gpu_id, p.pid)

    # ---- Wait for all workers ----
    for w in workers:
        w.join()
        if w.exitcode != 0:
            logger.error("Worker pid=%d exited with code %d", w.pid, w.exitcode)

    stop_event.set()
    monitor_thread.join(timeout=5)

    done = counter.value
    errors = error_counter.value
    logger.info(
        "ALL DONE — processed=%d errors=%d total=%d",
        done, errors, total,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch annotate with WhisperX — multi-GPU."
    )
    parser.add_argument("egs", type=Path, help="Path to egs jsonl[.gz] file, or a directory to scan recursively for .wav files")
    parser.add_argument("--lang", default="en", help="Language code.")
    parser.add_argument(
        "--whisper_model", default="large-v3",
        help="WhisperX model size (default: large-v3).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for WhisperX transcription (default: 16). Lower if OOM.",
    )
    parser.add_argument(
        "--rerun_errors", action="store_true",
        help="Ignore previous .err files and retry failed files.",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=4,
        help="Number of GPUs to use (default: 4).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    params = Params(
        egs=args.egs,
        verbose=args.verbose,
        lang=args.lang,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        rerun_errors=args.rerun_errors,
        num_gpus=args.num_gpus,
    )
    run(params)


if __name__ == "__main__":
    mp.set_start_method("spawn")  # required for CUDA in subprocesses
    main()
