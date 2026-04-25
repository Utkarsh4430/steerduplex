"""VoiceBench evaluation driver for SteerDuplex / Moshi-style models.

Runs the full 4-stage pipeline with per-stage resume:
  A. inference — Moshi generates audio per VoiceBench example (multi-GPU / multi-instance).
  B. ASR      — local Parakeet (subprocess into raman_fdb_v1 env; see asr_parakeet.py).
  C. judge    — parallel gpt-4o-mini judge for open/qa splits (see llm_judge.py).
  D. summary  — aggregate into summary.md + summary.json (see summarize.py).

Each stage skips records that are already complete, so re-running this script
just finishes whatever's missing and re-emits the summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from tqdm import tqdm

from eval.voicebench.splits import (
    SDQA_ALL_REGIONS,
    SDQA_DEFAULT_REGIONS,
    SUPPORTED_SUBSETS,
    SplitSpec,
    build_specs,
    unique_id_for,
)

DEFAULT_SYSTEM_PROMPT = "Answer the question concisely."
SANITY_LIMIT = 5

logger = logging.getLogger(__name__)

load_dotenv()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceBench inference for Moshi-style models")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for all outputs (per-split subdirs + summary.md).")
    parser.add_argument("--splits", nargs="+", default=list(SUPPORTED_SUBSETS),
                        help=f"Subsets to run. Default: all supported ({', '.join(SUPPORTED_SUBSETS)}).")
    parser.add_argument("--sdqa_regions", nargs="+", default=None,
                        help=f"sd-qa regions. Default: {SDQA_DEFAULT_REGIONS} (a warning is logged). "
                             f"Available: {', '.join(SDQA_ALL_REGIONS)}.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to finetuned checkpoint dir.")
    parser.add_argument("--hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--device", dest="devices", nargs="+", default=["cuda:0"],
                        help="Physical GPU device(s) to use.")
    parser.add_argument("--instances_per_gpu", type=int, default=2,
                        help="Model instances per GPU. Workers = len(devices) * instances_per_gpu.")
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--voice_prompt", type=str, default=None,
                        help="Path to voice-prompt WAV or .pt embeddings file "
                             "(conditioning audio applied once before the user turn).")
    parser.add_argument("--sanity", action="store_true",
                        help=f"Run only the first {SANITY_LIMIT} examples per split (smoke).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Per-split example cap (overrides --sanity if set).")
    parser.add_argument("--only", choices=["inference", "asr", "judge", "summary"],
                        default=None,
                        help="Run only one stage; default runs all stages that have pending work.")
    parser.add_argument("--skip_asr", action="store_true",
                        help="Skip Stage B (Parakeet ASR).")
    parser.add_argument("--skip_judge", action="store_true",
                        help="Skip Stage C (LLM judge).")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel workers for the LLM judge. Default: 32.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini",
                        help="LiteLLM model identifier for the LLM judge (open & qa evaluators).")
    parser.add_argument("--litellm-base", type=str, default=None,
                        help="LiteLLM proxy base URL (overrides LITELLM_BASE_URL).")
    parser.add_argument(
        "--asr_env_python", type=str,
        default="/mnt/efs/ramaneswaranselvakumar/miniconda3/envs/raman_fdb_v1/bin/python",
        help="Python binary for the Parakeet ASR subprocess (needs nemo_toolkit).",
    )
    parser.add_argument("--asr_device", type=str, default="cuda:0",
                        help="GPU device for Parakeet transcription.")
    parser.add_argument("--asr_batch_size", type=int, default=32,
                        help="Parakeet batch size.")
    parser.add_argument("--force_asr", action="store_true",
                        help="Re-transcribe every record from scratch (wipes scores too).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_split_dirs(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "output_audios").mkdir(parents=True, exist_ok=True)
    (split_dir / "_inputs").mkdir(parents=True, exist_ok=True)


def load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
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


def carry_row_fields(spec: SplitSpec, row: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the fields that should be preserved from the HF row into the output record."""
    out: Dict[str, Any] = {}
    for field in spec.carry_fields:
        if field in row and field != "audio":
            out[field] = row[field]
    return out


# ---------------------------------------------------------------------------
# Pre-extraction — write each dataset audio to a WAV on disk so workers can
# share it without paying per-example IPC costs.
# ---------------------------------------------------------------------------

def prepare_split_inputs(
    spec: SplitSpec,
    split_dir: Path,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """Load the HF dataset, write each audio to `_inputs/<uid>.wav`, and return
    a list of lightweight example records with `audio_path` set.

    We set `decode=False` on the Audio column and decode the raw bytes with
    soundfile directly. This avoids the torchcodec/ffmpeg dependency chain
    that ships with datasets>=3.0.
    """
    import io

    from datasets import Audio, load_dataset

    inputs_dir = split_dir / "_inputs"
    ds = load_dataset("hlt-lab/voicebench", spec.subset, split=spec.hf_split)
    ds = ds.cast_column("audio", Audio(decode=False))

    total = len(ds)
    if limit is not None:
        total = min(limit, total)

    records: List[Dict[str, Any]] = []
    for idx in range(total):
        row = ds[idx]
        uid = unique_id_for(spec, idx, row)
        audio_path = inputs_dir / f"{uid}.wav"
        if not audio_path.exists():
            audio_info = row["audio"]
            if audio_info.get("bytes") is not None:
                buf = io.BytesIO(audio_info["bytes"])
                array, sr = sf.read(buf, dtype="float32", always_2d=False)
            elif audio_info.get("path"):
                array, sr = sf.read(audio_info["path"], dtype="float32", always_2d=False)
            else:
                raise RuntimeError(f"No audio bytes/path for {uid}")
            # Mono-mix if stereo.
            if array.ndim == 2:
                array = array.mean(axis=1).astype(np.float32)
            sf.write(audio_path, array, sr)
        records.append({
            "unique_id": uid,
            "split": spec.name,
            "index": idx,
            "audio_path": str(audio_path),
            **carry_row_fields(spec, row),
        })
    return records


# ---------------------------------------------------------------------------
# Inference worker (one per model instance)
# ---------------------------------------------------------------------------

def _run_one(model, output_dir: Path, record: Dict[str, Any],
             system_prompt: str, voice_prompt: Optional[str],
             max_duration: float) -> Optional[Dict[str, Any]]:
    try:
        out_audio, out_sr, out_text_tokens = model.generate(
            user_audio_path=record["audio_path"],
            system_prompt=system_prompt,
            voice_prompt_path=voice_prompt,
            max_duration_sec=max_duration,
        )
    except Exception:
        logger.exception("Inference failed for %s", record["unique_id"])
        return None

    moshi_text = "".join(t for t in out_text_tokens if not t.startswith("["))

    rel_out = Path("output_audios") / f"{record['unique_id']}.wav"
    abs_out = output_dir / rel_out
    sf.write(abs_out, out_audio, out_sr)

    # Build output record — drop internal fields (audio_path, index).
    result = {k: v for k, v in record.items() if k not in ("audio_path", "index")}
    result["output_audio_path"] = rel_out.as_posix()
    result["moshi_text"] = moshi_text
    result["transcription"] = None
    result["score"] = None
    return result


def _inference_worker(
    worker_id: int,
    device: str,
    records: List[Dict[str, Any]],
    output_dir: Path,
    system_prompt: str,
    voice_prompt: Optional[str],
    max_duration: float,
    hf_repo: str,
    checkpoint: Optional[str],
    seed: int,
    greedy: bool,
    result_queue: mp.Queue,
) -> None:
    # Restrict to a single physical GPU before CUDA init. Multiple workers with
    # the same CUDA_VISIBLE_DEVICES will share that GPU without OOM on H100.
    import torch
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and device_obj.index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_obj.index)
        device = "cuda:0"

    setup_logging()

    def log(msg: str) -> None:
        result_queue.put(("log", f"[w{worker_id}:{device_obj}] {msg}"))

    log(f"Loading Moshi from {hf_repo} (assigned {len(records)} examples)...")
    try:
        from inference.generate import MoshiInference
        model = MoshiInference(
            hf_repo_id=hf_repo,
            checkpoint_path=checkpoint,
            device=device,
            seed=seed,
            greedy=greedy,
        )
    except Exception as exc:
        log(f"ERROR: model init failed: {exc}")
        result_queue.put(("done", worker_id))
        return

    log("Model loaded. Starting inference.")
    for record in records:
        result = _run_one(model, output_dir, record, system_prompt, voice_prompt, max_duration)
        if result is not None:
            result_queue.put(("result", result))

    log("Worker finished.")
    result_queue.put(("done", worker_id))


# ---------------------------------------------------------------------------
# Main per-split orchestration
# ---------------------------------------------------------------------------

def _build_worker_assignments(
    devices: List[str], instances_per_gpu: int
) -> List[Tuple[int, str]]:
    """Return list of (worker_id, device_str); length = len(devices) * instances_per_gpu."""
    assignments: List[Tuple[int, str]] = []
    wid = 0
    for device in devices:
        for _ in range(instances_per_gpu):
            assignments.append((wid, device))
            wid += 1
    return assignments


def run_inference_for_split(
    spec: SplitSpec,
    split_dir: Path,
    devices: List[str],
    instances_per_gpu: int,
    system_prompt: str,
    voice_prompt: Optional[str],
    max_duration: float,
    hf_repo: str,
    checkpoint: Optional[str],
    seed: int,
    greedy: bool,
    limit: Optional[int],
) -> None:
    ensure_split_dirs(split_dir)
    output_json = split_dir / "output.json"

    logger.info("Preparing inputs for %s...", spec.name)
    records = prepare_split_inputs(spec, split_dir, limit=limit)

    existing = load_existing_results(output_json)
    done = {r["unique_id"] for r in existing}
    pending = [r for r in records if r["unique_id"] not in done]

    logger.info(
        "Split %s — %d total, %d already done, %d pending.",
        spec.name, len(records), len(done), len(pending),
    )
    if not pending:
        return

    assignments = _build_worker_assignments(devices, instances_per_gpu)
    n_workers = len(assignments)

    # Single-worker fast path (no multiprocessing).
    if n_workers == 1:
        from inference.generate import MoshiInference
        worker_id, device = assignments[0]
        import torch
        device_obj = torch.device(device)
        if device_obj.type == "cuda" and device_obj.index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_obj.index)
            device = "cuda:0"
        model = MoshiInference(
            hf_repo_id=hf_repo, checkpoint_path=checkpoint,
            device=device, seed=seed, greedy=greedy,
        )
        pbar = tqdm(total=len(pending), desc=f"{spec.name}", unit="ex", dynamic_ncols=True)
        for idx, record in enumerate(pending):
            result = _run_one(model, split_dir, record, system_prompt, voice_prompt, max_duration)
            if result is not None:
                existing.append(result)
                if (idx + 1) % 10 == 0:
                    save_results(existing, output_json)
            pbar.update(1)
        save_results(existing, output_json)
        pbar.close()
        return

    logger.info("Pre-caching HF repo %s before spawning workers.", hf_repo)
    snapshot_download(repo_id=hf_repo)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    workers: List[mp.Process] = []
    for (wid, device), worker_records in zip(
        assignments,
        [pending[i::n_workers] for i in range(n_workers)],
    ):
        if not worker_records:
            continue
        p = ctx.Process(
            target=_inference_worker,
            args=(wid, device, worker_records, split_dir, system_prompt,
                  voice_prompt, max_duration, hf_repo, checkpoint, seed, greedy, result_queue),
        )
        p.start()
        workers.append(p)
        logger.info("Worker w%d on %s → %d examples", wid, device, len(worker_records))

    active = {i for i in range(n_workers)}
    save_interval = 10
    unsaved = 0
    pbar = tqdm(total=len(pending), desc=spec.name, unit="ex", dynamic_ncols=True)

    while active:
        try:
            kind, payload = result_queue.get(timeout=1.0)
            if kind == "log":
                tqdm.write(payload)
            elif kind == "result":
                if payload["unique_id"] in done:
                    continue
                existing.append(payload)
                done.add(payload["unique_id"])
                unsaved += 1
                pbar.update(1)
                if unsaved >= save_interval:
                    save_results(existing, output_json)
                    unsaved = 0
            elif kind == "done":
                active.discard(payload)
        except queue.Empty:
            pass

        for wid, p in enumerate(workers):
            if wid in active and p.exitcode not in (None, 0):
                logger.error("Worker %d exited with code %s", wid, p.exitcode)
                active.discard(wid)

    pbar.close()

    # Drain any remaining results before join (avoids pipe deadlock).
    while True:
        try:
            kind, payload = result_queue.get_nowait()
            if kind == "result" and payload["unique_id"] not in done:
                existing.append(payload)
                done.add(payload["unique_id"])
        except queue.Empty:
            break

    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            logger.warning("Worker did not exit cleanly; terminating.")
            p.terminate()

    save_results(existing, output_json)


# ---------------------------------------------------------------------------
# Entry point — orchestrates all splits, then Phase 2 and Phase 3.
# ---------------------------------------------------------------------------

def _resolve_sdqa_regions(explicit: Optional[List[str]]) -> List[str]:
    if explicit is not None:
        return explicit
    others = [r for r in SDQA_ALL_REGIONS if r not in SDQA_DEFAULT_REGIONS]
    logger.warning(
        "sd-qa restricted to %s region(s); %d other regions (%s) are skipped. "
        "Pass --sdqa_regions to evaluate additional regions.",
        ", ".join(SDQA_DEFAULT_REGIONS),
        len(others),
        ", ".join(others),
    )
    return list(SDQA_DEFAULT_REGIONS)


def _write_run_manifest(output_dir: Path, args: argparse.Namespace, specs: List[SplitSpec]) -> None:
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hf_repo": args.hf_repo,
        "checkpoint": args.checkpoint,
        "system_prompt": args.system_prompt,
        "voice_prompt": args.voice_prompt,
        "max_duration": args.max_duration,
        "greedy": args.greedy,
        "seed": args.seed,
        "devices": args.devices,
        "instances_per_gpu": args.instances_per_gpu,
        "sanity": args.sanity,
        "limit": args.limit,
        "splits": [{"name": s.name, "subset": s.subset, "hf_split": s.hf_split} for s in specs],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def _run_stage(name: str, only: Optional[str], skip: bool = False) -> bool:
    """Decide whether to execute a given stage under the --only / --skip gates."""
    if only is not None:
        return only == name
    return not skip


def main() -> None:
    args = parse_args()
    setup_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sdqa_regions = _resolve_sdqa_regions(args.sdqa_regions) if "sd-qa" in args.splits else []
    specs = build_specs(args.splits, sdqa_regions)

    logger.info("Output directory: %s", output_dir)
    logger.info("Running %d split unit(s):", len(specs))
    for s in specs:
        logger.info("  - %s (subset=%s, hf_split=%s, evaluator=%s)",
                    s.name, s.subset, s.hf_split, s.evaluator)

    limit = args.limit if args.limit is not None else (SANITY_LIMIT if args.sanity else None)
    if limit is not None:
        logger.info("Per-split limit: %d", limit)

    _write_run_manifest(output_dir, args, specs)

    # Stage A: inference. Per-split resume already lives in run_inference_for_split.
    if _run_stage("inference", args.only):
        t0 = time.time()
        for i, spec in enumerate(specs, 1):
            logger.info("[%d/%d] Stage A (inference): %s", i, len(specs), spec.name)
            split_dir = output_dir / spec.name
            run_inference_for_split(
                spec=spec,
                split_dir=split_dir,
                devices=args.devices,
                instances_per_gpu=args.instances_per_gpu,
                system_prompt=args.system_prompt,
                voice_prompt=args.voice_prompt,
                max_duration=args.max_duration,
                hf_repo=args.hf_repo,
                checkpoint=args.checkpoint,
                seed=args.seed,
                greedy=args.greedy,
                limit=limit,
            )
        logger.info("Stage A complete in %.1fs.", time.time() - t0)
    else:
        logger.info("Skipping Stage A (inference).")

    # Stage B: Parakeet ASR via subprocess into raman_fdb_v1 env.
    if _run_stage("asr", args.only, skip=args.skip_asr):
        from eval.voicebench import llm_judge
        llm_judge.run_parakeet_asr(
            output_dir=output_dir,
            specs=specs,
            asr_env_python=args.asr_env_python,
            asr_device=args.asr_device,
            batch_size=args.asr_batch_size,
            force=args.force_asr,
        )
    else:
        logger.info("Skipping Stage B (ASR).")

    # Stage C: LLM judge.
    if _run_stage("judge", args.only, skip=args.skip_judge):
        from eval.voicebench import llm_judge
        llm_judge.run_judge_pipeline(
            output_dir=output_dir,
            specs=specs,
            litellm_base=args.litellm_base,
            num_workers=args.workers,
            judge_model=args.judge_model,
        )
    else:
        logger.info("Skipping Stage C (judge).")

    # Stage D: summary. Always runs by default so markdown stays in sync.
    if _run_stage("summary", args.only):
        from eval.voicebench import summarize
        summarize.write_summary(output_dir=output_dir, specs=specs, run_args=vars(args))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
