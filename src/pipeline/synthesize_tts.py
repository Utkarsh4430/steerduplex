"""Phase 3: Synthesize speech using Qwen3-TTS.

Assistant turns: CustomVoice model (preset speaker + instruct for style control).
User turns: Base model (voice cloning via ref_audio, no style control).

Resumable: skips already-synthesized conversations and individual turns.
Multi-GPU: distributes conversations across GPUs with multiple workers per GPU
to maximize utilization (autoregressive TTS is memory-light but latency-bound).

Usage:
    python -m pipeline.synthesize_tts --config configs/generation.yaml
    python -m pipeline.synthesize_tts --config configs/generation.yaml --num_gpus 8 --workers_per_gpu 4
    python -m pipeline.synthesize_tts --config configs/generation.yaml --category A3_tone_controlled
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp

from pipeline.utils import ensure_dir, load_json, load_yaml, save_json, set_seed

# Suppress noisy HuggingFace warnings (pad_token_id, etc.)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)


import re

# Parenthetical markers that should never be spoken
_NON_SPOKEN_RE = re.compile(
    r"\s*\("
    r"(?:laughs?|sighs?|pauses?|chuckles?|clears?\s*throat|whispers?|gasps?|coughs?"
    r"|smiles?|nods?|grins?|winks?|sniffs?|sobs?|giggles?|groans?|mumbles?"
    r"|breathes?\s*(?:deeply|in|out)?|exhales?|inhales?)"
    r"\)\s*",
    re.IGNORECASE,
)

# Max comma-separated parts in an instruct string. More than this makes TTS overact.
_MAX_INSTRUCT_PARTS = 3


def clean_text_for_tts(text: str) -> str:
    """Strip non-spoken markers like (sighs), (laughs), (pauses) from text."""
    text = _NON_SPOKEN_RE.sub(" ", text)
    # Collapse multiple spaces / leading/trailing whitespace
    return re.sub(r"\s+", " ", text).strip()


def simplify_instruct(instruct) -> str | None:
    """Keep instruct short — max 3 comma-parts. TTS overacts with more.
    Handles cases where LLM generates a dict instead of a string."""
    if not instruct:
        return None
    # LLM sometimes generates {"emotion": "calm", "speed": "fast"} instead of a string
    if isinstance(instruct, dict):
        # Extract values, skip keys like "voice"/"rate"/"pitch" that aren't style descriptors
        parts = []
        for k, v in instruct.items():
            if k in ("voice", "rate", "pitch", "volume"):
                continue
            if isinstance(v, str) and v not in ("default", "neutral", "normal"):
                parts.append(v)
        instruct = ", ".join(parts) if parts else None
        if not instruct:
            return None
    if not isinstance(instruct, str):
        return None
    parts = [p.strip() for p in instruct.split(",") if p.strip()]
    if len(parts) <= _MAX_INSTRUCT_PARTS:
        return instruct
    return ", ".join(parts[:_MAX_INSTRUCT_PARTS])


class TTSSynthesizer:
    """Dual-model Qwen3-TTS wrapper."""

    def __init__(
        self,
        assistant_model_id: str,
        user_model_id: str,
        device: str = "cuda:0",
    ):
        # Suppress warnings during model load too
        import transformers
        transformers.logging.set_verbosity_error()

        from qwen_tts import Qwen3TTSModel

        self.device = device
        self.cv_model = Qwen3TTSModel.from_pretrained(
            assistant_model_id, device_map=device, dtype=torch.bfloat16,
        )
        self.base_model = Qwen3TTSModel.from_pretrained(
            user_model_id, device_map=device, dtype=torch.bfloat16,
        )
        self._user_prompts: dict[str, object] = {}

    def synthesize_assistant(
        self, text: str, speaker: str, instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        text = clean_text_for_tts(text)
        instruct = simplify_instruct(instruct)
        kwargs = dict(text=text, language="English", speaker=speaker)
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sr = self.cv_model.generate_custom_voice(**kwargs)
        audio = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio.astype(np.float32), sr

    def synthesize_user_clone(
        self, text: str, ref_audio: str, ref_text: str = "",
    ) -> tuple[np.ndarray, int]:
        text = clean_text_for_tts(text)
        if ref_audio not in self._user_prompts:
            self._user_prompts[ref_audio] = self.base_model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text or "Reference audio.",
                x_vector_only_mode=not bool(ref_text),
            )
        wavs, sr = self.base_model.generate_voice_clone(
            text=text, language="English",
            voice_clone_prompt=self._user_prompts[ref_audio],
        )
        audio = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio.astype(np.float32), sr

    def synthesize_user_preset(self, text: str, speaker: str) -> tuple[np.ndarray, int]:
        text = clean_text_for_tts(text)
        wavs, sr = self.cv_model.generate_custom_voice(
            text=text, language="English", speaker=speaker,
        )
        audio = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio.astype(np.float32), sr


def resample_if_needed(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=src_sr, target_sr=target_sr)


def synthesize_conversation(
    synth: TTSSynthesizer,
    transcript: dict,
    output_dir: Path,
    target_sr: int = 24000,
    max_retries: int = 3,
) -> dict | None:
    """Synthesize all turns of a conversation. Saves each turn as WAV immediately."""
    conv_id = transcript["id"]
    conv_dir = ensure_dir(output_dir / conv_id)

    assistant_voice = transcript.get("assistant_voice", {})
    user_voice = transcript.get("user_voice", {})

    updated_turns = []
    for i, turn in enumerate(transcript["turns"]):
        role = turn["role"]
        text = turn["text"]
        out_path = conv_dir / f"turn_{i:03d}_{role}.wav"

        # Resume: skip if already synthesized
        if out_path.exists():
            updated_turns.append({**turn, "audio_path": str(out_path)})
            continue

        success = False
        for retry in range(max_retries):
            try:
                if role == "assistant":
                    audio, sr = synth.synthesize_assistant(
                        text=text,
                        speaker=assistant_voice.get("speaker", "Ryan"),
                        instruct=turn.get("tts_instruct"),
                    )
                else:
                    if user_voice.get("model") == "Base" and user_voice.get("ref_path"):
                        audio, sr = synth.synthesize_user_clone(
                            text=text,
                            ref_audio=user_voice["ref_path"],
                            ref_text=user_voice.get("ref_text", ""),
                        )
                    else:
                        audio, sr = synth.synthesize_user_preset(
                            text=text,
                            speaker=user_voice.get("speaker", "Eric"),
                        )

                audio = resample_if_needed(audio, sr, target_sr)
                peak = np.abs(audio).max()
                if peak > 0:
                    audio = audio * (0.95 / peak)
                sf.write(str(out_path), audio, target_sr)

                updated_turns.append({**turn, "audio_path": str(out_path)})
                success = True
                break

            except Exception as e:
                print(f"  [{synth.device} RETRY {retry + 1}] {conv_id} turn {i} ({role}): {e}")

        if not success:
            print(f"  [{synth.device} FAIL] {conv_id} turn {i}")
            return None

    return {**transcript, "turns": updated_turns}


# ---------------------------------------------------------------------------
# Shared progress counter for cross-process tqdm
# ---------------------------------------------------------------------------
_progress_counter: mp.Value = None
_fail_counter: mp.Value = None


def _update_progress(success: bool = True):
    """Atomically increment the shared progress counter."""
    if success:
        if _progress_counter is not None:
            with _progress_counter.get_lock():
                _progress_counter.value += 1
    else:
        if _fail_counter is not None:
            with _fail_counter.get_lock():
                _fail_counter.value += 1


# ---------------------------------------------------------------------------
# Multi-GPU, multi-worker
# ---------------------------------------------------------------------------
def _worker_fn(
    worker_id: int,
    total_workers: int,
    gpu_id: int,
    work_items: list[tuple[Path, Path, Path]],
    assistant_model_id: str,
    user_model_id: str,
    target_sr: int,
    max_retries: int,
    seed: int,
    progress_counter,
    fail_counter,
):
    """Worker process: loads models on assigned GPU and processes its shard."""
    from pipeline.distributed import is_done, release_claim, try_claim

    # Set up shared counters in this process
    global _progress_counter, _fail_counter
    _progress_counter = progress_counter
    _fail_counter = fail_counter

    # Suppress warnings in worker process
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

    device = f"cuda:{gpu_id}"
    set_seed(seed + worker_id)

    # Round-robin partition across ALL workers
    my_items = work_items[worker_id::total_workers]
    if not my_items:
        return

    tag = f"W{worker_id}/GPU{gpu_id}"
    print(f"[{tag}] Loading models on {device}, {len(my_items)} conversations", flush=True)

    synth = TTSSynthesizer(
        assistant_model_id=assistant_model_id,
        user_model_id=user_model_id,
        device=device,
    )
    print(f"[{tag}] Models loaded, starting synthesis", flush=True)

    for transcript_path, synth_out_path, claim_path in my_items:
        # Skip if already done (another node/worker may have finished it)
        if is_done(synth_out_path):
            _update_progress(success=True)
            continue

        # Atomic claim — skip if another worker (local or remote) claimed it
        if not try_claim(claim_path):
            continue

        transcript = load_json(transcript_path)
        result = synthesize_conversation(
            synth, transcript, synth_out_path.parent,
            target_sr=target_sr,
            max_retries=max_retries,
        )
        if result:
            save_json(result, synth_out_path)
            release_claim(claim_path)
            _update_progress(success=True)
        else:
            release_claim(claim_path)  # let another worker retry
            _update_progress(success=False)


def _progress_monitor(total: int, progress_counter, fail_counter):
    """Monitor thread that drives a tqdm bar from shared counters."""
    from tqdm import tqdm

    pbar = tqdm(total=total, desc="Synthesizing", unit="conv",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    last = 0
    while True:
        with progress_counter.get_lock():
            done = progress_counter.value
        with fail_counter.get_lock():
            failed = fail_counter.value
        current = done + failed

        if current > last:
            pbar.update(current - last)
            pbar.set_postfix(ok=done, fail=failed, refresh=False)
            last = current

        if current >= total:
            break
        time.sleep(0.5)

    pbar.close()


def _collect_work(
    assignments_dir: Path,
    output_dir: Path,
    category: str | None,
) -> list[tuple[Path, Path, Path]]:
    """Collect all (transcript_path, synth_output_path, claim_path) triples needing processing."""
    from pipeline.distributed import is_done

    work_items = []
    claims_base = output_dir / ".claims"

    for cat_dir in sorted(assignments_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if category and cat_dir.name != category:
            continue

        cat_output = ensure_dir(output_dir / cat_dir.name)
        cat_claims = ensure_dir(claims_base / cat_dir.name)
        transcripts = sorted(cat_dir.glob("*.json"))

        done_count = 0
        for transcript_path in transcripts:
            transcript = load_json(transcript_path)
            conv_id = transcript["id"]
            synth_out_path = cat_output / f"{conv_id}_synth.json"
            claim_path = cat_claims / f"{conv_id}.claim"
            if is_done(synth_out_path):
                done_count += 1
            else:
                work_items.append((transcript_path, synth_out_path, claim_path))

        remaining_count = len(transcripts) - done_count
        if remaining_count == 0:
            print(f"[SKIP] {cat_dir.name}: all {len(transcripts)} done")
        else:
            print(f"[QUEUE] {cat_dir.name}: {remaining_count} remaining ({done_count} done)")

    return work_items


def main():
    parser = argparse.ArgumentParser(description="Synthesize TTS audio")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--assignments_dir", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs (default: all available)")
    parser.add_argument("--workers_per_gpu", type=int, default=None,
                        help="Concurrent workers per GPU (default: from config or 4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["tts"]
    set_seed(args.seed)

    assignments_dir = Path(args.assignments_dir or load_yaml(args.config)["voices"]["output_dir"])
    output_dir = ensure_dir(cfg["output_dir"])
    target_sr = cfg.get("sample_rate", 24000)
    max_retries = cfg.get("max_retries", 3)

    # Pre-download models to HF cache before spawning workers (avoids download races)
    print("Pre-caching TTS models...")
    from huggingface_hub import snapshot_download
    for model_id in [cfg["assistant_model_id"], cfg["user_model_id"]]:
        snapshot_download(model_id)
    print("Models cached.\n")

    # Collect remaining work and shuffle so different nodes don't race on the same items
    import random as _random
    work_items = _collect_work(assignments_dir, output_dir, args.category)
    _random.shuffle(work_items)

    if not work_items:
        print("Nothing to synthesize.")
        return

    # GPU memory-aware worker planning
    # Qwen3-TTS: ~3.4GB per model x 2 models = ~6.8GB weights
    # + KV cache grows to ~2-3GB on long sequences + PyTorch allocator overhead
    # Peak observed: ~10GB per worker. Use 11GB for safety.
    from pipeline.distributed import plan_workers

    TTS_MEM_PER_WORKER_MB = 11000
    max_wpg = args.workers_per_gpu or cfg.get("workers_per_gpu", 8)
    req_gpus = args.num_gpus or cfg.get("num_gpus") or None

    print("Planning GPU workers based on available memory:")
    gpu_plans = plan_workers(
        mem_per_worker_mb=TTS_MEM_PER_WORKER_MB,
        max_workers_per_gpu=max_wpg,
        min_free_after_mb=4096,
        num_gpus=req_gpus,
    )

    if not gpu_plans:
        print("[ERROR] No GPUs with enough free memory. Need at least "
              f"{TTS_MEM_PER_WORKER_MB + 2048}MB free per GPU.")
        sys.exit(1)

    total_workers = min(sum(p.num_workers for p in gpu_plans), len(work_items))
    num_gpus = len(gpu_plans)
    print(f"\nTotal: {len(work_items)} conversations | {num_gpus} GPUs, {total_workers} workers\n")

    if total_workers == 1:
        # Single worker — use tqdm directly
        from tqdm import tqdm
        import transformers
        from pipeline.distributed import is_done, release_claim, try_claim
        transformers.logging.set_verbosity_error()

        device = f"cuda:{gpu_plans[0].gpu_id}" if gpu_plans else "cuda:0"
        synth = TTSSynthesizer(
            assistant_model_id=cfg["assistant_model_id"],
            user_model_id=cfg["user_model_id"],
            device=device,
        )

        done = 0
        failed = 0
        for transcript_path, synth_out_path, claim_path in tqdm(work_items, desc="Synthesizing", unit="conv"):
            if is_done(synth_out_path):
                done += 1
                continue
            if not try_claim(claim_path):
                continue  # another node/process claimed it
            transcript = load_json(transcript_path)
            result = synthesize_conversation(
                synth, transcript, synth_out_path.parent,
                target_sr=target_sr, max_retries=max_retries,
            )
            if result:
                save_json(result, synth_out_path)
                release_claim(claim_path)
                done += 1
            else:
                release_claim(claim_path)
                failed += 1

        print(f"\nDone: {done}, Failed: {failed}")
    else:
        # Multi-worker with shared progress bar
        mp.set_start_method("spawn", force=True)

        progress_counter = mp.Value("i", 0)
        fail_counter = mp.Value("i", 0)

        # Start progress monitor thread
        import threading
        monitor = threading.Thread(
            target=_progress_monitor,
            args=(len(work_items), progress_counter, fail_counter),
            daemon=True,
        )
        monitor.start()

        # Launch workers — assign based on memory-aware gpu_plans
        processes = []
        global_worker_id = 0
        for plan in gpu_plans:
            for _ in range(plan.num_workers):
                p = mp.Process(
                    target=_worker_fn,
                    args=(
                        global_worker_id, total_workers, plan.gpu_id,
                        work_items,
                        cfg["assistant_model_id"],
                        cfg["user_model_id"],
                        target_sr,
                        max_retries,
                        args.seed,
                        progress_counter,
                        fail_counter,
                    ),
                )
                p.start()
                processes.append(p)
                global_worker_id += 1

        for p in processes:
            p.join()

        # Let monitor finish
        monitor.join(timeout=5)

        failed_workers = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed_workers:
            print(f"\n[WARN] Workers {failed_workers} exited with errors")

        with progress_counter.get_lock():
            done = progress_counter.value
        with fail_counter.get_lock():
            failed = fail_counter.value
        print(f"\nDone: {done}, Failed: {failed}")

    print("Synthesis complete.")


if __name__ == "__main__":
    main()
