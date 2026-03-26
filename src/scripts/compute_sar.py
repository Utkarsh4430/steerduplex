"""Compute Speech Activity Ratio (SAR) for training datasets.

SAR = (frames where assistant channel has speech) / (total frames after prompt region)

High SAR (>0.5): assistant is active — good for turn-taking training
Low SAR (<0.2): assistant is mostly silent — teaches passivity

Usage:
    # Just compute distribution (no changes)
    python scripts/compute_sar.py --manifest data/formatted/manifest_combined_train.jsonl

    # Compute and filter (create new manifest excluding low-SAR files)
    python scripts/compute_sar.py --manifest data/external/fisher_prompted/manifest.jsonl \
        --min_sar 0.15 --output data/external/fisher_prompted/manifest_filtered.jsonl

    # Compute for specific dataset
    python scripts/compute_sar.py --audio_dir data/external/annutacon_prompted/audio --sample 500
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)

SPEECH_RMS_THRESHOLD = 0.01  # RMS above this = speech
FRAME_DURATION_MS = 100       # 100ms analysis frames


def compute_sar(wav_path: str, prompt_end_sec: float = 0.0) -> dict:
    """Compute Speech Activity Ratio for a stereo WAV file.

    Returns dict with:
        sar: float (0-1), speech activity ratio of assistant channel
        assistant_speech_pct: float, percentage of time assistant speaks
        user_speech_pct: float, percentage of time user speaks
        duration_sec: float, total duration
        prompt_end_sec: float
    """
    try:
        audio, sr = sf.read(wav_path)
    except Exception as e:
        return {"error": str(e)}

    if audio.ndim != 2 or audio.shape[1] < 2:
        return {"error": "not stereo"}

    # Skip prompt region
    start_sample = int(prompt_end_sec * sr)
    if start_sample >= audio.shape[0]:
        return {"error": "prompt longer than audio"}

    ch0 = audio[start_sample:, 0]  # assistant
    ch1 = audio[start_sample:, 1]  # user

    frame_size = int(FRAME_DURATION_MS / 1000 * sr)
    n_frames = len(ch0) // frame_size

    if n_frames == 0:
        return {"error": "too short"}

    asst_speech = 0
    user_speech = 0

    for i in range(n_frames):
        s = i * frame_size
        e = s + frame_size
        if np.sqrt(np.mean(ch0[s:e] ** 2)) > SPEECH_RMS_THRESHOLD:
            asst_speech += 1
        if np.sqrt(np.mean(ch1[s:e] ** 2)) > SPEECH_RMS_THRESHOLD:
            user_speech += 1

    return {
        "sar": asst_speech / n_frames,
        "assistant_speech_pct": asst_speech / n_frames,
        "user_speech_pct": user_speech / n_frames,
        "duration_sec": audio.shape[0] / sr,
        "prompt_end_sec": prompt_end_sec,
        "n_frames": n_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Speech Activity Ratio (SAR)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="JSONL manifest to analyze")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory of WAV files to analyze")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N files (default: all)")
    parser.add_argument("--min_sar", type=float, default=None,
                        help="If set, filter out files below this SAR threshold")
    parser.add_argument("--output", type=str, default=None,
                        help="Output manifest path (required if --min_sar is set)")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    if args.min_sar is not None and args.output is None:
        parser.error("--output required when using --min_sar")

    # Collect files to analyze
    files = []
    if args.manifest:
        manifest_dir = Path(args.manifest).parent
        with open(args.manifest) as f:
            entries = [json.loads(l) for l in f]
        for e in entries:
            wav_path = manifest_dir / e["path"]
            # Try to get prompt_end_sec from JSON
            json_path = wav_path.with_suffix(".json")
            prompt_end = 0.0
            if json_path.exists():
                try:
                    jdata = json.load(open(json_path))
                    tc = jdata.get("text_conditions", {})
                    prompt_end = float(tc.get("prompt_end_sec", 0))
                except:
                    pass
            files.append({"path": str(wav_path), "prompt_end": prompt_end, "entry": e})
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        for wav in sorted(audio_dir.glob("*.wav")):
            json_path = wav.with_suffix(".json")
            prompt_end = 0.0
            if json_path.exists():
                try:
                    jdata = json.load(open(json_path))
                    tc = jdata.get("text_conditions", {})
                    prompt_end = float(tc.get("prompt_end_sec", 0))
                except:
                    pass
            files.append({"path": str(wav), "prompt_end": prompt_end})
    else:
        parser.error("Either --manifest or --audio_dir required")

    if args.sample and len(files) > args.sample:
        import random
        random.seed(42)
        files = random.sample(files, args.sample)

    logger.info("Analyzing %d files...", len(files))

    # Compute SAR using multiprocessing.Pool (CPU-bound RMS computation)
    import multiprocessing as mp

    results = []
    work_items = [(i, f["path"], f["prompt_end"]) for i, f in enumerate(files)]

    def _sar_worker(args):
        idx, path, prompt_end = args
        result = compute_sar(path, prompt_end)
        result["_idx"] = idx
        return result

    with mp.Pool(processes=args.num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_sar_worker, work_items, chunksize=64),
            total=len(work_items), desc="Computing SAR",
        ):
            idx = result.pop("_idx")
            result["path"] = files[idx]["path"]
            if "entry" in files[idx]:
                result["entry"] = files[idx]["entry"]
            results.append(result)

    # Filter errors
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    logger.info("Valid: %d, Errors: %d", len(valid), len(errors))

    if not valid:
        logger.error("No valid results")
        return

    sars = [r["sar"] for r in valid]

    # Print distribution
    print(f"\n{'='*60}")
    print(f"SAR DISTRIBUTION ({len(valid)} files)")
    print(f"{'='*60}")
    print(f"  Mean:   {np.mean(sars):.3f}")
    print(f"  Median: {np.median(sars):.3f}")
    print(f"  Std:    {np.std(sars):.3f}")
    print(f"  Min:    {np.min(sars):.3f}")
    print(f"  Max:    {np.max(sars):.3f}")
    print()

    # Histogram
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0]
    counts, _ = np.histogram(sars, bins=bins)
    print(f"  {'SAR Range':15s} {'Count':>8s} {'Pct':>6s} {'Cumul':>6s}")
    print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*6}")
    cumul = 0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        cumul += counts[i]
        pct = counts[i] / len(sars) * 100
        cumul_pct = cumul / len(sars) * 100
        bar = "█" * int(pct / 2)
        print(f"  {lo:.2f} - {hi:.2f}    {counts[i]:>8d} {pct:>5.1f}% {cumul_pct:>5.1f}% {bar}")

    # Dataset-level breakdown if manifest
    if args.manifest:
        # Group by source
        from collections import defaultdict
        source_sars = defaultdict(list)
        for r in valid:
            path = r["path"]
            if "annutacon_studio" in path:
                src = "annutacon_studio"
            elif "annutacon_remaining" in path:
                src = "annutacon_remaining"
            elif "annutacon" in path:
                src = "annutacon"
            elif "fisher" in path:
                src = "fisher"
            else:
                name = Path(path).stem
                src = "synthetic"
            source_sars[src].append(r["sar"])

        print(f"\n  Per-source SAR:")
        for src in sorted(source_sars, key=lambda x: np.mean(source_sars[x])):
            vals = source_sars[src]
            below_20 = sum(1 for v in vals if v < 0.2) / len(vals) * 100
            print(f"    {src:25s} mean={np.mean(vals):.3f}  median={np.median(vals):.3f}  <0.2={below_20:.0f}%  (n={len(vals)})")

    # Filter if requested
    if args.min_sar is not None:
        kept = [r for r in valid if r["sar"] >= args.min_sar and "entry" in r]
        removed = len(valid) - len(kept)
        logger.info("Filtering: keeping %d/%d (removed %d with SAR < %.2f)",
                     len(kept), len(valid), removed, args.min_sar)

        with open(args.output, "w") as f:
            for r in kept:
                f.write(json.dumps(r["entry"]) + "\n")
        logger.info("Written filtered manifest: %s", args.output)

        kept_hours = sum(r["entry"].get("duration", 0) for r in kept) / 3600
        removed_hours = sum(r["entry"].get("duration", 0) for r in valid if r["sar"] < args.min_sar and "entry" in r) / 3600
        print(f"\n  Kept: {len(kept)} files ({kept_hours:.0f}h)")
        print(f"  Removed: {removed} files ({removed_hours:.0f}h)")


if __name__ == "__main__":
    main()
