"""Quality filtering for synthesized audio (lightweight for scale).

Checks:
1. WER via Whisper ASR — is the speech intelligible?
2. Duration — is the turn too short or too long?
3. Silence ratio — is it mostly silence?

Multi-GPU: spawns workers across GPUs, each loads its own Whisper model.
Whisper medium ≈ 1.5 GB per worker, so 40+ workers fit on an 80GB GPU.
Default: all available GPUs, 8 workers per GPU.

Distributed: uses atomic file claiming — safe to run on N nodes simultaneously.

Usage:
    python -m pipeline.quality_filter --config configs/generation.yaml
    python -m pipeline.quality_filter --config configs/generation.yaml --num_gpus 8 --workers_per_gpu 8
    python -m pipeline.quality_filter --config configs/generation.yaml --category A3_tone_controlled
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp

from pipeline.utils import load_json, load_yaml, save_json

# Suppress noisy warnings
logging.getLogger("whisper").setLevel(logging.ERROR)


class QualityChecker:
    def __init__(self, wer_threshold: float = 0.20, min_duration: float = 0.5, max_silence_ratio: float = 0.5, whisper_model: str = "medium", device: str = "cuda:0"):
        self.wer_threshold = wer_threshold
        self.min_duration = min_duration
        self.max_silence_ratio = max_silence_ratio
        self.whisper_model_name = whisper_model
        self.device = device
        self._whisper = None

    @property
    def whisper(self):
        if self._whisper is None:
            import whisper
            self._whisper = whisper.load_model(self.whisper_model_name, device=self.device)
        return self._whisper

    def check_wer(self, audio_path: str, expected_text: str) -> tuple[float, bool]:
        from jiwer import wer
        result = self.whisper.transcribe(audio_path, language="en")
        transcribed = result["text"].strip().lower()
        expected = expected_text.strip().lower()
        for marker in ["(laughs)", "(sighs)", "(pauses)", "(clears throat)", "(whispers)"]:
            expected = expected.replace(marker, "").strip()
        if not expected:
            return 0.0, True
        error_rate = wer(expected, transcribed)
        return round(error_rate, 3), error_rate <= self.wer_threshold

    def check_duration(self, audio_path: str) -> tuple[float, bool]:
        info = sf.info(audio_path)
        return round(info.duration, 2), info.duration >= self.min_duration

    def check_silence(self, audio_path: str) -> tuple[float, bool]:
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        energy = np.abs(audio)
        silence_frames = (energy < 0.01).sum()
        ratio = silence_frames / max(len(audio), 1)
        return round(float(ratio), 3), ratio <= self.max_silence_ratio

    def check_turn(self, audio_path: str, expected_text: str) -> dict:
        results = {"passed": True, "checks": {}}

        dur, dur_ok = self.check_duration(audio_path)
        results["checks"]["duration"] = {"value": float(dur), "passed": bool(dur_ok)}
        if not dur_ok:
            results["passed"] = False
            return results  # skip expensive WER if too short

        sil, sil_ok = self.check_silence(audio_path)
        results["checks"]["silence_ratio"] = {"value": float(sil), "passed": bool(sil_ok)}
        if not sil_ok:
            results["passed"] = False

        wer_score, wer_ok = self.check_wer(audio_path, expected_text)
        results["checks"]["wer"] = {"value": float(wer_score), "passed": bool(wer_ok)}
        if not wer_ok:
            results["passed"] = False

        return results


def filter_conversation(checker: QualityChecker, transcript: dict) -> dict:
    all_passed = True
    for turn in transcript["turns"]:
        audio_path = turn.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            turn["quality_check"] = {"passed": False, "reason": "missing_audio"}
            all_passed = False
            continue

        check = checker.check_turn(audio_path, turn["text"])
        turn["quality_check"] = check
        if not check["passed"]:
            all_passed = False

    transcript["quality_passed"] = bool(all_passed)
    return transcript


# ---------------------------------------------------------------------------
# Shared progress counter for cross-process tqdm
# ---------------------------------------------------------------------------
_progress_counter: mp.Value = None
_pass_counter: mp.Value = None


def _update_progress(passed: bool):
    if _progress_counter is not None:
        with _progress_counter.get_lock():
            _progress_counter.value += 1
    if passed and _pass_counter is not None:
        with _pass_counter.get_lock():
            _pass_counter.value += 1


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------
def _worker_fn(
    worker_id: int,
    total_workers: int,
    gpu_id: int,
    work_items: list[tuple[Path, Path]],
    wer_threshold: float,
    min_duration: float,
    max_silence_ratio: float,
    whisper_model: str,
    progress_counter,
    pass_counter,
):
    from pipeline.distributed import is_done, release_claim, try_claim

    global _progress_counter, _pass_counter
    _progress_counter = progress_counter
    _pass_counter = pass_counter

    device = f"cuda:{gpu_id}"

    # Round-robin partition
    my_items = work_items[worker_id::total_workers]
    if not my_items:
        return

    tag = f"QF-W{worker_id}/GPU{gpu_id}"
    print(f"[{tag}] Loading Whisper on {device}, {len(my_items)} conversations", flush=True)

    checker = QualityChecker(
        wer_threshold=wer_threshold,
        min_duration=min_duration,
        max_silence_ratio=max_silence_ratio,
        whisper_model=whisper_model,
        device=device,
    )
    # Force load now
    _ = checker.whisper
    print(f"[{tag}] Whisper loaded, starting", flush=True)

    for synth_path, claim_path in my_items:
        transcript = load_json(synth_path)
        if "quality_passed" in transcript:
            _update_progress(transcript["quality_passed"])
            continue

        if not try_claim(claim_path):
            continue

        transcript = filter_conversation(checker, transcript)
        save_json(transcript, synth_path)
        release_claim(claim_path)
        _update_progress(transcript["quality_passed"])


def _progress_monitor(total: int, progress_counter, pass_counter):
    from tqdm import tqdm
    pbar = tqdm(total=total, desc="Quality filter", unit="conv",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    last = 0
    while True:
        with progress_counter.get_lock():
            done = progress_counter.value
        with pass_counter.get_lock():
            passed = pass_counter.value
        if done > last:
            pbar.update(done - last)
            pbar.set_postfix(passed=passed, failed=done - passed, refresh=False)
            last = done
        if done >= total:
            break
        time.sleep(0.5)
    pbar.close()


def _collect_work(synth_dir: Path, category: str | None) -> list[tuple[Path, Path]]:
    """Collect (synth_path, claim_path) pairs needing quality filtering."""
    work_items = []
    claims_base = synth_dir / ".claims_qf"

    for cat_dir in sorted(synth_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("."):
            continue
        if category and cat_dir.name != category:
            continue

        cat_claims = claims_base / cat_dir.name
        cat_claims.mkdir(parents=True, exist_ok=True)

        done = 0
        todo = 0
        for synth_path in sorted(cat_dir.glob("*_synth.json")):
            transcript = load_json(synth_path)
            if "quality_passed" in transcript:
                done += 1
            else:
                claim_path = cat_claims / f"{synth_path.stem}.claim"
                work_items.append((synth_path, claim_path))
                todo += 1

        if todo == 0:
            print(f"[SKIP] {cat_dir.name}: all {done} checked")
        else:
            print(f"[QUEUE] {cat_dir.name}: {todo} remaining ({done} done)")

    return work_items


def main():
    parser = argparse.ArgumentParser(description="Quality filter synthesized audio")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--synth_dir", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs (default: all available)")
    parser.add_argument("--workers_per_gpu", type=int, default=None,
                        help="Whisper workers per GPU (default: 8, ~1.5GB each)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)["quality"]

    synth_dir = Path(args.synth_dir or load_yaml(args.config)["tts"]["output_dir"])

    work_items = _collect_work(synth_dir, args.category)
    random.shuffle(work_items)

    if not work_items:
        print("Nothing to filter.")
        return

    # GPU memory-aware worker planning
    # Whisper medium ≈ 1.5GB per worker
    from pipeline.distributed import plan_workers

    WHISPER_MEM_PER_WORKER_MB = 2500  # 1.5GB model + KV cache + overhead
    max_wpg = args.workers_per_gpu or 8
    req_gpus = args.num_gpus or None

    print("Planning GPU workers based on available memory:")
    gpu_plans = plan_workers(
        mem_per_worker_mb=WHISPER_MEM_PER_WORKER_MB,
        max_workers_per_gpu=max_wpg,
        min_free_after_mb=2048,
        num_gpus=req_gpus,
    )

    if not gpu_plans:
        print("[WARN] No GPUs available. Running on CPU (slow).")
        gpu_plans = []  # will fall through to single-worker CPU path

    total_workers = min(sum(p.num_workers for p in gpu_plans), len(work_items)) if gpu_plans else 1
    print(f"\nTotal: {len(work_items)} conversations | {len(gpu_plans)} GPUs, {total_workers} workers\n")

    if total_workers <= 1:
        from pipeline.distributed import release_claim, try_claim
        from tqdm import tqdm

        device = f"cuda:{gpu_plans[0].gpu_id}" if gpu_plans else "cpu"
        checker = QualityChecker(
            wer_threshold=cfg.get("wer_threshold", 0.20),
            min_duration=cfg.get("min_duration_sec", 0.5),
            max_silence_ratio=cfg.get("max_silence_ratio", 0.5),
            whisper_model=cfg.get("whisper_model", "medium"),
            device=device,
        )
        passed = 0
        total = 0
        for synth_path, claim_path in tqdm(work_items, desc="Quality filter"):
            transcript = load_json(synth_path)
            if "quality_passed" in transcript:
                total += 1
                if transcript["quality_passed"]:
                    passed += 1
                continue
            if not try_claim(claim_path):
                continue
            transcript = filter_conversation(checker, transcript)
            save_json(transcript, synth_path)
            release_claim(claim_path)
            total += 1
            if transcript["quality_passed"]:
                passed += 1
        print(f"\nQuality: {passed}/{total} passed ({passed/max(total,1)*100:.1f}%)")
    else:
        mp.set_start_method("spawn", force=True)
        progress_counter = mp.Value("i", 0)
        pass_counter = mp.Value("i", 0)

        import threading
        monitor = threading.Thread(
            target=_progress_monitor,
            args=(len(work_items), progress_counter, pass_counter),
            daemon=True,
        )
        monitor.start()

        processes = []
        global_worker_id = 0
        for plan in gpu_plans:
            for _ in range(plan.num_workers):
                p = mp.Process(
                    target=_worker_fn,
                    args=(
                        global_worker_id, total_workers, plan.gpu_id,
                        work_items,
                        cfg.get("wer_threshold", 0.20),
                        cfg.get("min_duration_sec", 0.5),
                        cfg.get("max_silence_ratio", 0.5),
                        cfg.get("whisper_model", "medium"),
                        progress_counter,
                        pass_counter,
                    ),
                )
                p.start()
                processes.append(p)
                global_worker_id += 1

        for p in processes:
            p.join()

        monitor.join(timeout=5)

        with progress_counter.get_lock():
            total = progress_counter.value
        with pass_counter.get_lock():
            passed = pass_counter.value
        print(f"\nQuality: {passed}/{total} passed ({passed/max(total,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
