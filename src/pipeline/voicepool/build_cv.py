"""
Build a voice pool JSON from a Common Voice corpus directory.

Strategy (efficient — no filesystem enumeration of clips/):
  1. Read validated.tsv (or train/dev/test) — all metadata lives here.
  2. Optionally read clip_durations.tsv for pre-computed durations.
  3. Group clips by client_id (= speaker).
  4. For each speaker, pick the best clip near target duration.
  5. Convert that single MP3 → WAV and copy to output audios/ dir.

Common Voice layout:
  {lang}/clips/                        # flat dir, hundreds of thousands of MP3s
  {lang}/validated.tsv                 # columns: client_id, path, sentence, ...
  {lang}/clip_durations.tsv            # columns: clip, duration[ms]
  {lang}/train.tsv, dev.tsv, test.tsv  # subsets of validated

Output JSON format (matches LibriSpeech pool):
  {
    "id": "<speaker_hash>",
    "ref_path": "audios/<speaker_hash>.wav",
    "ref_text": "transcript",
    "original_path": "clips/common_voice_en_XXXXX.mp3",
    "duration": 4.12,
    "gender": "male" | "female" | null,
    "accent": "us" | "england" | ... | null,
    "age_range": "twenties" | ... | null,
    "energy": null
  }
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import subprocess
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Hardcoded paths ──────────────────────────────────────────────────────────
CV_BASE_DIR = Path(
    "/fs/nexus-projects/brain_project/ashish/speech_data/common_voice/"
    "common_voice_new/cv-corpus-18.0-2024-06-14/en"
)
OUTPUT_DIR = Path(
    "/fs/gamma-projects/audio/raman/steerd/steerduplex/src/data/voices/user"
)

# ── Defaults ─────────────────────────────────────────────────────────────────
TARGET_DUR = 4.0     # seconds
TOLERANCE = 2.0      # seconds — accept clips in [target-tol, target+tol]
DEFAULT_WORKERS = 8  # parallel ffmpeg conversions


def load_clip_durations(cv_dir: Path) -> dict[str, float]:
    """Load clip_durations.tsv → {filename: duration_seconds}."""
    dur_path = cv_dir / "clip_durations.tsv"
    durations: dict[str, float] = {}
    if not dur_path.exists():
        log.warning("clip_durations.tsv not found — will probe MP3s for duration (slower)")
        return durations

    log.info("Loading clip_durations.tsv ...")
    t0 = time.time()
    with open(dur_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Common Voice stores duration in milliseconds
            clip_name = row.get("clip") or row.get("clip_name") or ""
            dur_raw = row.get("duration") or row.get("duration[ms]") or "0"
            try:
                dur_ms = float(dur_raw)
                # Heuristic: if value > 1000 it's probably ms, otherwise seconds
                if dur_ms > 1000:
                    durations[clip_name] = dur_ms / 1000.0
                else:
                    durations[clip_name] = dur_ms
            except ValueError:
                continue

    elapsed = time.time() - t0
    log.info("  Loaded %d clip durations in %.1fs", len(durations), elapsed)
    return durations


def probe_duration_ffprobe(mp3_path: Path) -> float | None:
    """Get duration via ffprobe (fallback if clip_durations.tsv missing)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(mp3_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def build_split_map(cv_dir: Path) -> dict[str, str]:
    """Read train.tsv, dev.tsv, test.tsv and build {clip_path: split_name}."""
    split_map: dict[str, str] = {}
    for split_name, tsv_name in [("train", "train.tsv"), ("dev", "dev.tsv"), ("test", "test.tsv")]:
        tsv_path = cv_dir / tsv_name
        if not tsv_path.exists():
            log.warning("Split TSV not found: %s", tsv_path)
            continue
        t0 = time.time()
        count = 0
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                clip_path = row.get("path", "").strip()
                if clip_path:
                    split_map[clip_path] = split_name
                    count += 1
        log.info("  %s.tsv: %d clips (%.1fs)", split_name, count, time.time() - t0)
    log.info("Split map built: %d clips mapped to train/dev/test", len(split_map))
    return split_map


def load_tsv_and_group_by_speaker(
    cv_dir: Path,
    tsv_name: str,
    clip_durations: dict[str, float],
    split_map: dict[str, str],
) -> dict[str, list[dict]]:
    """
    Read a TSV file, attach duration info and split label, group rows by client_id (speaker).
    Returns {client_id: [list of clip dicts]}.
    """
    tsv_path = cv_dir / tsv_name
    if not tsv_path.exists():
        log.error("TSV not found: %s", tsv_path)
        sys.exit(1)

    log.info("Reading %s ...", tsv_name)
    t0 = time.time()

    speakers: dict[str, list[dict]] = defaultdict(list)
    total_rows = 0
    rows_with_duration = 0

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            client_id = row.get("client_id", "").strip()
            clip_path = row.get("path", "").strip()
            sentence = row.get("sentence", "").strip()
            gender = row.get("gender", "").strip()
            age = row.get("age", "").strip()
            accent = row.get("accent", "").strip()

            if not client_id or not clip_path:
                continue

            # Attach duration from precomputed table
            dur = clip_durations.get(clip_path)
            if dur is not None:
                rows_with_duration += 1

            speakers[client_id].append({
                "path": clip_path,
                "sentence": sentence,
                "duration": dur,  # may be None
                "gender": gender if gender else None,
                "age": age if age else None,
                "accent": accent if accent else None,
                "split": split_map.get(clip_path),  # "train", "dev", "test", or None
            })

    elapsed = time.time() - t0
    log.info(
        "  Parsed %d rows, %d unique speakers in %.1fs  (%.0f%% have precomputed duration)",
        total_rows, len(speakers), elapsed,
        100.0 * rows_with_duration / max(total_rows, 1),
    )
    return dict(speakers)


def pick_best_clip(
    clips: list[dict],
    target_dur: float,
    tolerance: float,
) -> dict | None:
    """
    From a speaker's clip list, pick the one closest to target_dur.
    Prefer clips that have a known duration and a non-empty transcript.
    """
    # First pass: only clips with known duration
    candidates = [c for c in clips if c["duration"] is not None and c["sentence"]]
    if not candidates:
        # Relax: allow missing transcript
        candidates = [c for c in clips if c["duration"] is not None]
    if not candidates:
        # Last resort: return first clip with a transcript (duration unknown)
        with_text = [c for c in clips if c["sentence"]]
        return with_text[0] if with_text else (clips[0] if clips else None)

    # Sort by distance to target
    candidates.sort(key=lambda c: abs(c["duration"] - target_dur))

    best = candidates[0]
    dist = abs(best["duration"] - target_dur)

    # Prefer within tolerance, but always return best available
    if dist <= tolerance:
        return best
    # Still return it as fallback
    return best


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """Convert MP3 → WAV (16kHz mono 16-bit) via ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp3_path),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                str(wav_path),
            ],
            capture_output=True, timeout=30,
        )
        return wav_path.exists()
    except Exception as e:
        log.warning("ffmpeg failed for %s: %s", mp3_path.name, e)
        return False


def normalize_gender(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.strip().lower()
    if r in ("male", "male_masculine", "masculine"):
        return "male"
    if r in ("female", "female_feminine", "feminine"):
        return "female"
    if r in ("other", "other_gender"):
        return "other"
    return None


def normalize_age(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.strip().lower().replace(" ", "")
    # Common Voice uses: teens, twenties, thirties, forties, fifties, sixties, seventies, eighties, nineties
    if r in ("teens", "twenties", "thirties", "forties", "fifties",
             "sixties", "seventies", "eighties", "nineties"):
        return r
    return None


def fmt_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m"


def process_one_speaker(
    args_tuple: tuple[str, list[dict]],
    clips_dir: Path,
    audios_dir: Path,
    target_dur: float,
    tolerance: float,
) -> dict | None:
    """
    Worker function: pick best clip for one speaker, convert MP3→WAV.
    Returns a pool entry dict or None.
    """
    spk_id, clips = args_tuple

    best = pick_best_clip(clips, target_dur, tolerance)
    if best is None:
        return None

    mp3_path = clips_dir / best["path"]
    if not mp3_path.exists():
        return None

    # If duration was unknown, probe it
    dur = best["duration"]
    if dur is None:
        dur = probe_duration_ffprobe(mp3_path)
        if dur is None:
            dur = 0.0

    # Convert MP3 → WAV
    wav_name = f"{spk_id}.wav"
    wav_dest = audios_dir / wav_name

    if not wav_dest.exists():
        ok = convert_mp3_to_wav(mp3_path, wav_dest)
        if not ok:
            return None

    return {
        "id": spk_id,
        "ref_path": f"audios/{wav_name}",
        "ref_text": best["sentence"] if best["sentence"] else None,
        "original_path": f"clips/{best['path']}",
        "duration": round(dur, 2),
        "split": best.get("split"),
        "gender": normalize_gender(best["gender"]),
        "accent": best["accent"],
        "age_range": normalize_age(best["age"]),
        "energy": None,
    }


def build_voice_pool(
    cv_dir: Path,
    speakers: dict[str, list[dict]],
    output_dir: Path,
    target_dur: float,
    tolerance: float,
    num_workers: int = DEFAULT_WORKERS,
) -> list[dict]:
    """
    For each speaker, pick the best clip, convert MP3→WAV, build pool entry.
    Uses multiprocessing for parallel ffmpeg conversions.
    Logs progress with ETA.
    """
    clips_dir = cv_dir / "clips"
    audios_dir = output_dir / "audios"
    audios_dir.mkdir(parents=True, exist_ok=True)

    sorted_speakers = sorted(speakers.items())
    total = len(sorted_speakers)

    log.info("Processing %d speakers with %d workers ...", total, num_workers)
    log.info("-" * 60)

    # Create worker with fixed args via partial
    worker_fn = partial(
        process_one_speaker,
        clips_dir=clips_dir,
        audios_dir=audios_dir,
        target_dur=target_dur,
        tolerance=tolerance,
    )

    pool_entries = []
    t_start = time.time()
    completed = 0
    skipped = 0

    with mp.Pool(processes=num_workers) as proc_pool:
        # imap_unordered for best throughput + progress tracking
        for result in proc_pool.imap_unordered(worker_fn, sorted_speakers, chunksize=32):
            completed += 1
            if result is not None:
                pool_entries.append(result)
            else:
                skipped += 1

            # ── Progress + ETA ────────────────────────────────────────
            if completed % 200 == 0 or completed == total:
                elapsed = time.time() - t_start
                rate = completed / elapsed
                remaining = (total - completed) / rate if rate > 0 else 0
                pct = 100.0 * completed / total
                log.info(
                    "  [%5d / %d]  %5.1f%%  |  %.1f spk/s  |  elapsed %s  |  ETA %s  |  pool: %d  skipped: %d",
                    completed, total, pct, rate, fmt_eta(elapsed), fmt_eta(remaining),
                    len(pool_entries), skipped,
                )

    elapsed_total = time.time() - t_start
    log.info("-" * 60)
    log.info("Done in %s  |  %d speakers in pool, %d skipped", fmt_eta(elapsed_total), len(pool_entries), skipped)
    return pool_entries


def main():
    parser = argparse.ArgumentParser(
        description="Build Common Voice voice pool JSON (efficient, speaker-first)."
    )
    parser.add_argument(
        "--cv_dir", type=str, default=str(CV_BASE_DIR),
        help="Common Voice language directory (default: hardcoded EN path)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for JSON + audios/ (default: hardcoded path)",
    )
    parser.add_argument(
        "--tsv", type=str, default="validated.tsv",
        help="Which TSV to use as source (default: validated.tsv)",
    )
    parser.add_argument(
        "--target_dur", type=float, default=TARGET_DUR,
        help="Target clip duration in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=TOLERANCE,
        help="Acceptable deviation from target (default: 2.0)",
    )
    parser.add_argument(
        "--max_speakers", type=int, default=None,
        help="Limit number of speakers (for testing). Default: all.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel workers for ffmpeg conversion (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cv_dir = Path(args.cv_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Common Voice Pool Builder")
    log.info("=" * 60)
    log.info("CV directory : %s", cv_dir)
    log.info("Output dir   : %s", output_dir)
    log.info("Source TSV   : %s", args.tsv)
    log.info("Target dur   : %.1fs ± %.1fs", args.target_dur, args.tolerance)
    log.info("Workers      : %d", args.workers)
    if args.max_speakers:
        log.info("Max speakers : %d (test mode)", args.max_speakers)
    log.info("=" * 60)

    # ── Step 1: Load precomputed durations ────────────────────────────────
    clip_durations = load_clip_durations(cv_dir)

    # ── Step 2: Build clip → split mapping from train/dev/test TSVs ───────
    log.info("Building split map from train/dev/test TSVs ...")
    split_map = build_split_map(cv_dir)

    # ── Step 3: Read TSV, group by speaker ────────────────────────────────
    speakers = load_tsv_and_group_by_speaker(cv_dir, args.tsv, clip_durations, split_map)

    if args.max_speakers:
        # Trim for testing
        trimmed = dict(list(speakers.items())[:args.max_speakers])
        log.info("Trimmed to %d speakers (from %d) for testing", len(trimmed), len(speakers))
        speakers = trimmed

    # ── Step 3: Summary before processing ─────────────────────────────────
    total_clips = sum(len(v) for v in speakers.values())
    clips_per_spk = total_clips / max(len(speakers), 1)
    log.info("Speakers: %d  |  Total clips: %d  |  Avg clips/speaker: %.1f",
             len(speakers), total_clips, clips_per_spk)

    # Accent/gender overview from metadata
    gender_counts: dict[str, int] = defaultdict(int)
    accent_counts: dict[str, int] = defaultdict(int)
    for spk_clips in speakers.values():
        rep = spk_clips[0]
        g = normalize_gender(rep["gender"])
        gender_counts[str(g)] += 1
        if rep["accent"]:
            accent_counts[rep["accent"]] += 1

    log.info("Gender distribution (all speakers): %s",
             ", ".join(f"{k}: {v}" for k, v in sorted(gender_counts.items())))
    if accent_counts:
        log.info("Top accents: %s",
                 ", ".join(f"{k}: {v}" for k, v in sorted(accent_counts.items(), key=lambda x: -x[1])[:10]))

    # Filter to only speakers with male/female gender
    before_filter = len(speakers)
    speakers = {
        spk_id: clips for spk_id, clips in speakers.items()
        if normalize_gender(clips[0]["gender"]) in ("male", "female")
    }
    log.info("Filtered to male/female speakers: %d -> %d (dropped %d with unknown gender)",
             before_filter, len(speakers), before_filter - len(speakers))

    # ── Step 4: Build pool ────────────────────────────────────────────────
    pool = build_voice_pool(cv_dir, speakers, output_dir, args.target_dur, args.tolerance, args.workers)

    # ── Step 5: Save JSON ─────────────────────────────────────────────────
    json_path = output_dir / "cv_voice_pool.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    log.info("Saved pool JSON: %s", json_path)

    # ── Final summary ─────────────────────────────────────────────────────
    if pool:
        durations = [e["duration"] for e in pool]
        accents = defaultdict(int)
        ages = defaultdict(int)
        splits = defaultdict(int)
        genders = defaultdict(int)
        for e in pool:
            splits[str(e["split"])] += 1
            genders[str(e["gender"])] += 1
            if e["accent"]:
                accents[e["accent"]] += 1
            if e["age_range"]:
                ages[e["age_range"]] += 1

        log.info("=" * 60)
        log.info("FINAL POOL SUMMARY")
        log.info("=" * 60)
        log.info("Total voices     : %d", len(pool))
        log.info("Duration range   : %.2fs – %.2fs (mean %.2fs)",
                 min(durations), max(durations), sum(durations) / len(durations))
        log.info("Split breakdown  : %s",
                 ", ".join(f"{k}: {v}" for k, v in sorted(splits.items())))
        log.info("Gender breakdown : %s",
                 ", ".join(f"{k}: {v}" for k, v in sorted(genders.items())))
        log.info("Top accents      : %s",
                 ", ".join(f"{k}: {v}" for k, v in sorted(accents.items(), key=lambda x: -x[1])[:10]))
        log.info("Age distribution : %s",
                 ", ".join(f"{k}: {v}" for k, v in sorted(ages.items(), key=lambda x: -x[1])))
        log.info("JSON saved to    : %s", json_path)
        log.info("Audio files in   : %s", output_dir / "audios")

    log.info("Done!")


if __name__ == "__main__":
    main()