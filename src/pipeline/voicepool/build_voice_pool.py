"""
Build a voice pool JSON from a standard LibriSpeech directory.

Strategy:
  1. Discover unique speakers from directory structure.
  2. For each speaker, walk their files and stop at the first clip
     meeting the target duration (default ~4s).
  3. Compute duration from file size (avoids opening every WAV).

LibriSpeech layout:
  {split}/{speaker}/{chapter}/{speaker}-{chapter}-{utterance}.wav
  {split}/{speaker}/{chapter}/{speaker}-{chapter}.trans.txt
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# LibriSpeech is always 16kHz, 16-bit (2 bytes), mono
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
WAV_HEADER_SIZE = 44

SPLIT_MAP = {
    "train-clean-100": "train",
    "train-clean-360": "train",
    "train-other-500": "train",
    "dev-clean": "val",
    "dev-other": "val",
    "test-clean": "test",
    "test-other": "test",
}


def duration_from_filesize(path: Path) -> float:
    """Compute WAV duration from file size (16kHz 16-bit mono)."""
    size = path.stat().st_size
    return (size - WAV_HEADER_SIZE) / (BYTES_PER_SAMPLE * SAMPLE_RATE)


def parse_trans_txt(trans_path: Path) -> dict[str, str]:
    """Parse a .trans.txt file into {utterance_id: transcript}."""
    mapping = {}
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: <utterance_id> <transcript...>
            utt_id, _, transcript = line.partition(" ")
            mapping[utt_id] = transcript
    log.debug("Parsed %d transcripts from %s", len(mapping), trans_path.name)
    return mapping


def parse_speakers_txt(path: Path) -> dict[str, dict]:
    """Parse a SPEAKERS.txt file into {speaker_id: {gender: ...}}."""
    info = {}
    if not path.exists():
        log.warning("SPEAKERS.txt not found at %s", path)
        return info
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                spk_id = parts[0]
                g = parts[1].upper()
                info[spk_id] = {"gender": "Female" if g == "F" else "Male" if g == "M" else "Unknown"}
    log.info("Parsed %d speakers from %s", len(info), path)
    return info


def find_clip_for_speaker(
    speaker_dir: Path,
    base_dir: Path,
    target_dur: float = 4.0,
    tolerance: float = 2.0,
) -> tuple[dict, Path] | None:
    """
    Walk a speaker's chapters and return the first clip within
    [target_dur - tolerance, target_dur + tolerance].
    Falls back to the closest-duration clip if none match exactly.
    Returns (clip_info, absolute_wav_path) or None.
    """
    best_clip = None
    best_wav = None
    best_distance = float("inf")
    files_checked = 0
    chapters_checked = 0

    for chapter_dir in speaker_dir.iterdir():
        if not chapter_dir.is_dir():
            continue
        chapters_checked += 1

        # Parse chapter-level trans.txt once (LibriSpeech style)
        trans_file = next(chapter_dir.glob("*.trans.txt"), None)
        transcripts = parse_trans_txt(trans_file) if trans_file else {}

        for wav in chapter_dir.glob("*.wav"):
            files_checked += 1
            dur = duration_from_filesize(wav)
            distance = abs(dur - target_dur)

            # Try per-file transcript first (LibriTTS: {utt}.normalized.txt / {utt}.original.txt)
            transcript = ""
            norm_txt = wav.with_suffix(".normalized.txt")
            orig_txt = wav.with_suffix(".original.txt")
            if norm_txt.exists():
                transcript = norm_txt.read_text(encoding="utf-8").strip()
            elif orig_txt.exists():
                transcript = orig_txt.read_text(encoding="utf-8").strip()
            else:
                # Fall back to chapter-level .trans.txt (LibriSpeech)
                # LibriSpeech uses hyphens in trans.txt but filenames may use underscores
                utt_id = wav.stem
                transcript = transcripts.get(utt_id, "")
                if not transcript:
                    transcript = transcripts.get(utt_id.replace("_", "-"), "")

            clip = {
                "rel_path": str(wav.relative_to(base_dir)),
                "transcript": transcript,
                "duration": round(dur, 2),
            }

            # Early exit: within tolerance
            if distance <= tolerance:
                log.debug(
                    "Speaker %s: early exit after %d files/%d chapters — %.2fs clip",
                    speaker_dir.name, files_checked, chapters_checked, dur,
                )
                return clip, wav

            # Track best fallback
            if distance < best_distance:
                best_distance = distance
                best_clip = clip
                best_wav = wav

    if best_clip:
        log.debug(
            "Speaker %s: no clip in tolerance, fallback to %.2fs (checked %d files/%d chapters)",
            speaker_dir.name, best_clip["duration"], files_checked, chapters_checked,
        )
    else:
        log.warning("Speaker %s: no WAV files found", speaker_dir.name)

    return (best_clip, best_wav) if best_clip else None


def build_voice_pool(
    base_dir: Path,
    split_dirs: list[Path],
    speaker_meta: dict[str, dict],
    target_dur: float,
    tolerance: float,
    audios_dir: Path,
) -> list[dict]:
    """Build the voice pool for a set of split directories.
    
    Copies selected audio clips into audios_dir, renamed as {speaker_id}.wav.
    """
    # Collect unique speaker dirs across all split dirs for this split
    speaker_dirs: dict[str, Path] = {}
    for split_dir in tqdm(split_dirs, desc="Scanning split dirs"):
        if not split_dir.is_dir():
            log.warning("Split dir not found: %s", split_dir)
            continue
        for spk_dir in split_dir.iterdir():
            if spk_dir.is_dir():
                speaker_dirs[spk_dir.name] = spk_dir

    log.info("Found %d unique speakers across %d split dirs", len(speaker_dirs), len(split_dirs))

    pool = []
    skipped = 0
    fallback_count = 0
    copied_count = 0

    for spk_id, spk_dir in tqdm(sorted(speaker_dirs.items()), desc="Finding clips", unit="speaker"):
        result = find_clip_for_speaker(spk_dir, base_dir, target_dur, tolerance)
        if result is None:
            skipped += 1
            continue

        clip, abs_wav_path = result

        # Track how many used fallback (outside tolerance)
        if abs(clip["duration"] - target_dur) > tolerance:
            fallback_count += 1

        # Copy audio into audios/ as {speaker_id}.wav
        dest_wav = audios_dir / f"{spk_id}.wav"
        if not dest_wav.exists():
            shutil.copy2(abs_wav_path, dest_wav)
            copied_count += 1
            log.debug("Copied %s -> %s", abs_wav_path.name, dest_wav.name)

        meta = speaker_meta.get(spk_id, {})
        gender_raw = meta.get("gender", None)
        # Normalize to lowercase for output format, None if unknown
        gender_out = gender_raw.lower() if gender_raw and gender_raw != "Unknown" else None

        pool.append({
            "id": spk_id,
            "ref_path": f"audios/{spk_id}.wav",
            "ref_text": clip["transcript"] if clip["transcript"] else None,
            "original_path": clip["rel_path"],
            "duration": clip["duration"],
            "gender": gender_out,
            "accent": None,
            "age_range": None,
            "energy": None,
        })

    log.info(
        "Pool built: %d speakers, %d skipped (no clips), %d used fallback duration, %d audio files copied",
        len(pool), skipped, fallback_count, copied_count,
    )
    return pool


def main():
    parser = argparse.ArgumentParser(description="Build LibriSpeech voice pool JSON (fast, early-exit).")
    parser.add_argument("--base_dir", type=str, required=True, help="Root dir containing LibriSpeech splits")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSON files")
    parser.add_argument("--target_dur", type=float, default=4.0, help="Target clip duration in seconds")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Acceptable deviation from target (seconds)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (per-speaker detail)")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Base directory: %s", base_dir)
    log.info("Output directory: %s", out_dir)
    log.info("Target duration: %.1fs ± %.1fs", args.target_dur, args.tolerance)

    # Group top-level subdirs by split
    log.info("Discovering splits...")
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and entry.name in SPLIT_MAP:
            split_name = SPLIT_MAP[entry.name]
            splits[split_name].append(entry)
            log.info("  Found: %s -> %s", entry.name, split_name)

    # Load speaker metadata from SPEAKERS.txt in each known split dir (no rglob)
    log.info("Loading SPEAKERS.txt from split directories...")
    speaker_meta: dict[str, dict] = {}
    spk_files_found = 0
    for split_dir_list in splits.values():
        for split_dir in split_dir_list:
            spk_txt = split_dir / "SPEAKERS.txt"
            if spk_txt.exists():
                speaker_meta.update(parse_speakers_txt(spk_txt))
                spk_files_found += 1
    # Also check base dir itself
    if (base_dir / "SPEAKERS.txt").exists():
        speaker_meta.update(parse_speakers_txt(base_dir / "SPEAKERS.txt"))
        spk_files_found += 1
    log.info("Total speaker metadata loaded: %d speakers from %d files", len(speaker_meta), spk_files_found)

    # Build pool per split
    for split_name, split_dirs in splits.items():
        if not split_dirs:
            log.info("No directories found for '%s' split, skipping.", split_name)
            continue

        log.info("=" * 50)
        log.info("Building %s pool from %d source dirs...", split_name.upper(), len(split_dirs))

        # Create audios/ subdir for this split
        split_out_dir = out_dir / split_name
        audios_dir = split_out_dir / "audios"
        audios_dir.mkdir(parents=True, exist_ok=True)
        log.info("Audio files will be copied to: %s", audios_dir)

        pool = build_voice_pool(base_dir, split_dirs, speaker_meta, args.target_dur, args.tolerance, audios_dir)

        out_path = split_out_dir / f"libri_{split_name}_pool.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2)

        # Summary stats
        durations = [entry["duration"] for entry in pool]
        genders = {entry["gender"] for entry in pool}
        gender_counts = {g: sum(1 for e in pool if e["gender"] == g) for g in genders}

        log.info("Saved %s — %d speakers", out_path.name, len(pool))
        log.info(
            "  Duration range: %.2fs – %.2fs (mean %.2fs)",
            min(durations) if durations else 0,
            max(durations) if durations else 0,
            sum(durations) / len(durations) if durations else 0,
        )
        log.info("  Gender breakdown: %s", ", ".join(f"{g}: {n}" for g, n in sorted(gender_counts.items(), key=str)))

    log.info("Done!")


if __name__ == "__main__":
    main()