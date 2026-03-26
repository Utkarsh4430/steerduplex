"""Split and trim training audio for optimal chunking.

Problems this solves:
1. Long conversations (>163.84s) get arbitrarily chunked mid-turn by moshi-finetune
2. 84% of conversations end with assistant turn — model never learns "user spoke → respond"

What this does:
1. Split long audio at turn boundaries into chunks ≤ max_duration_sec
2. Every chunk ends with a user turn (assistant's last turn trimmed if needed)
3. System prompt region preserved in first chunk only
4. Produces new WAV + meta JSON pairs ready for format_dataset

Each output chunk gets:
- Stereo audio (ch0=assistant, ch1=user) with proper turn boundaries
- Updated meta JSON with correct turn_timestamps and prompt_end_sec
- Alignment JSON (text_conditions) matching the new audio

Usage:
    # Analyze only (report stats, no changes)
    python -m pipeline.split_audio --config configs/generation.yaml --dry_run

    # Process all assembled data
    python -m pipeline.split_audio --config configs/generation.yaml --num_workers 128

    # Process specific category
    python -m pipeline.split_audio --config configs/generation.yaml --category B9_duplex_patterns
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from pipeline.distributed import try_claim, release_claim, is_done
from pipeline.utils import ensure_dir, load_json, load_yaml, save_json

logger = logging.getLogger(__name__)

# Must match training config
MAX_DURATION_SEC = 163.84
SAMPLE_RATE = 24000
# Minimum useful chunk duration (too short = not enough context)
MIN_CHUNK_SEC = 10.0


def split_conversation(
    wav_path: str,
    meta: dict,
    max_duration_sec: float = MAX_DURATION_SEC,
) -> list[tuple[np.ndarray, dict]]:
    """Split a conversation into chunks that end with user turns.

    Returns list of (stereo_audio, updated_meta) tuples.
    """
    audio, sr = sf.read(wav_path)
    if audio.ndim != 2:
        return []

    turns = meta.get("turn_timestamps", [])
    prompt_end_sec = meta.get("prompt_end_sec", 0.0)
    prompt_end_samples = int(prompt_end_sec * sr)
    total_samples = audio.shape[0]

    if not turns:
        return []

    # Find split points: end of each user turn
    # Each chunk should end after a user turn (before the next assistant turn)
    user_turn_ends = []  # (turn_index, end_sample)
    for i, turn in enumerate(turns):
        if turn["role"] == "user":
            end_sample = int(turn["end_sec"] * sr)
            user_turn_ends.append((i, end_sample))

    if not user_turn_ends:
        # No user turns — skip this conversation
        return []

    # Build chunks
    chunks = []
    chunk_start_sample = 0  # First chunk includes prompt region
    chunk_start_turn_idx = 0
    is_first_chunk = True

    for ut_idx, (turn_idx, end_sample) in enumerate(user_turn_ends):
        chunk_duration = (end_sample - chunk_start_sample) / sr

        # Add a small silence buffer after user turn (200ms)
        buffer_samples = int(0.2 * sr)
        chunk_end_sample = min(end_sample + buffer_samples, total_samples)

        # Check if adding more turns would exceed max duration
        next_would_exceed = True
        if ut_idx + 1 < len(user_turn_ends):
            next_end = user_turn_ends[ut_idx + 1][1]
            if (next_end - chunk_start_sample) / sr <= max_duration_sec:
                next_would_exceed = False

        # Create chunk if: at max duration, or last user turn, or next would exceed
        if chunk_duration >= max_duration_sec * 0.8 or next_would_exceed or ut_idx == len(user_turn_ends) - 1:
            if chunk_duration < MIN_CHUNK_SEC:
                continue

            chunk_audio = audio[chunk_start_sample:chunk_end_sample]

            # Build turn timestamps relative to this chunk
            chunk_turns = []
            for t in turns[chunk_start_turn_idx:turn_idx + 1]:
                start = t["start_sec"] - chunk_start_sample / sr
                end = t["end_sec"] - chunk_start_sample / sr
                if start >= 0 and end > start:
                    chunk_turns.append({
                        "role": t["role"],
                        "start_sec": round(max(0, start), 3),
                        "end_sec": round(end, 3),
                        "text": t.get("text", ""),
                    })

            # Build chunk meta
            chunk_meta = {
                "id": meta["id"],
                "category": meta.get("category", ""),
                "data_type": meta.get("data_type", "standard"),
                "duration_sec": round(chunk_audio.shape[0] / sr, 3),
                "prompt_end_sec": round(prompt_end_sec, 3) if is_first_chunk else 0.0,
                "num_turns": len(chunk_turns),
                "turn_timestamps": chunk_turns,
                "system_prompt": meta.get("system_prompt", "") if is_first_chunk else "",
                "system_prompt_rephrased": meta.get("system_prompt_rephrased", "") if is_first_chunk else "",
                "assistant_voice_id": meta.get("assistant_voice_id", ""),
                "user_voice_id": meta.get("user_voice_id", ""),
            }

            chunks.append((chunk_audio, chunk_meta))

            # Next chunk starts after this user turn
            chunk_start_sample = chunk_end_sample
            chunk_start_turn_idx = turn_idx + 1
            is_first_chunk = False

    return chunks


def process_file(
    wav_path: Path,
    meta_path: Path,
    output_dir: Path,
    max_duration_sec: float = MAX_DURATION_SEC,
) -> list[dict]:
    """Process one conversation file. Returns list of output manifest entries."""
    meta = load_json(meta_path)
    chunks = split_conversation(str(wav_path), meta, max_duration_sec)

    entries = []
    for chunk_idx, (chunk_audio, chunk_meta) in enumerate(chunks):
        # Output file names
        base_id = meta.get("id", wav_path.stem)
        if len(chunks) == 1:
            out_id = base_id
        else:
            out_id = f"{base_id}_chunk{chunk_idx:02d}"

        chunk_meta["id"] = out_id
        out_wav = output_dir / f"{out_id}.wav"
        out_meta = output_dir / f"{out_id}_meta.json"

        if out_wav.exists() and out_meta.exists():
            entries.append({
                "id": out_id,
                "duration_sec": chunk_meta["duration_sec"],
                "last_turn_role": chunk_meta["turn_timestamps"][-1]["role"] if chunk_meta["turn_timestamps"] else "unknown",
            })
            continue

        # Normalize audio
        for ch in range(chunk_audio.shape[1]):
            peak = np.abs(chunk_audio[:, ch]).max()
            if peak > 0:
                chunk_audio[:, ch] *= 0.95 / peak

        sf.write(str(out_wav), chunk_audio, SAMPLE_RATE)
        save_json(chunk_meta, out_meta)

        entries.append({
            "id": out_id,
            "duration_sec": chunk_meta["duration_sec"],
            "last_turn_role": chunk_meta["turn_timestamps"][-1]["role"] if chunk_meta["turn_timestamps"] else "unknown",
        })

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Split and trim training audio for optimal chunking",
    )
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--input_dir", default=None,
                        help="Input assembled dir (default: from config)")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir (default: data/assembled_split)")
    parser.add_argument("--max_duration", type=float, default=MAX_DURATION_SEC)
    parser.add_argument("--category", default=None)
    parser.add_argument("--num_workers", type=int, default=128)
    parser.add_argument("--dry_run", action="store_true",
                        help="Report stats only, no processing")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(args.seed)

    cfg = load_yaml(args.config)
    input_dir = Path(args.input_dir or cfg.get("assembly", {}).get("output_dir", "data/assembled"))
    output_dir = Path(args.output_dir or "data/assembled_split")

    if not input_dir.exists():
        logger.error("Input dir not found: %s", input_dir)
        return

    # Find all WAV + meta pairs
    meta_files = sorted(input_dir.glob("*_meta.json"))
    if args.category:
        meta_files = [m for m in meta_files if args.category in m.name]

    logger.info("Found %d conversations in %s", len(meta_files), input_dir)

    if args.dry_run:
        # Analyze without processing
        stats = {
            "total": len(meta_files),
            "needs_split": 0,
            "ends_assistant": 0,
            "ends_user": 0,
            "already_good": 0,
            "durations": [],
        }

        for meta_path in tqdm(meta_files[:min(5000, len(meta_files))], desc="Analyzing"):
            meta = load_json(meta_path)
            dur = meta.get("duration_sec", 0)
            turns = meta.get("turn_timestamps", [])
            stats["durations"].append(dur)

            if dur > args.max_duration:
                stats["needs_split"] += 1

            if turns:
                if turns[-1]["role"] == "assistant":
                    stats["ends_assistant"] += 1
                else:
                    stats["ends_user"] += 1

            if dur <= args.max_duration and turns and turns[-1]["role"] == "user":
                stats["already_good"] += 1

        durs = np.array(stats["durations"])
        analyzed = len(stats["durations"])
        print(f"\n{'='*60}")
        print(f"SPLIT ANALYSIS ({analyzed} conversations)")
        print(f"{'='*60}")
        print(f"  Duration: mean={np.mean(durs):.0f}s  median={np.median(durs):.0f}s  max={np.max(durs):.0f}s")
        print(f"  Needs split (>{args.max_duration:.0f}s): {stats['needs_split']} ({stats['needs_split']/analyzed*100:.0f}%)")
        print(f"  Ends with assistant: {stats['ends_assistant']} ({stats['ends_assistant']/analyzed*100:.0f}%)")
        print(f"  Ends with user: {stats['ends_user']} ({stats['ends_user']/analyzed*100:.0f}%)")
        print(f"  Already good (short + ends user): {stats['already_good']} ({stats['already_good']/analyzed*100:.0f}%)")
        return

    # Process
    ensure_dir(output_dir)
    claims_dir = ensure_dir(output_dir / ".claims")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_entries = []
    to_process = []

    for meta_path in meta_files:
        wav_path = meta_path.with_name(meta_path.name.replace("_meta.json", ".wav"))
        if not wav_path.exists():
            continue
        to_process.append((wav_path, meta_path))

    logger.info("Processing %d files with %d workers...", len(to_process), args.num_workers)

    def _worker(item):
        wav_path, meta_path = item
        claim_path = claims_dir / f"{wav_path.stem}.claim"
        if not try_claim(claim_path):
            return []
        try:
            entries = process_file(wav_path, meta_path, output_dir, args.max_duration)
            return entries
        except Exception as e:
            logger.warning("Failed %s: %s", wav_path.name, e)
            release_claim(claim_path)
            return []

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_worker, item): item for item in to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Splitting"):
            entries = future.result()
            all_entries.extend(entries)

    # Stats
    user_end = sum(1 for e in all_entries if e["last_turn_role"] == "user")
    asst_end = sum(1 for e in all_entries if e["last_turn_role"] == "assistant")
    total_hours = sum(e["duration_sec"] for e in all_entries) / 3600

    print(f"\n{'='*60}")
    print(f"SPLIT RESULTS")
    print(f"{'='*60}")
    print(f"  Input: {len(to_process)} conversations")
    print(f"  Output: {len(all_entries)} chunks ({total_hours:.0f}h)")
    print(f"  Ends with user: {user_end} ({user_end/max(len(all_entries),1)*100:.0f}%)")
    print(f"  Ends with assistant: {asst_end} ({asst_end/max(len(all_entries),1)*100:.0f}%)")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
