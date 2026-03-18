"""
Build a voice pool JSON from VoxPopuli English ASR data (train split).

Strategy:
  1. Download asr_en.tsv.gz annotation file from Facebook CDN (cached locally).
  2. Filter to train split only.
  3. Group utterances by speaker_id.
  4. For each speaker, pick one utterance with duration closest to target (3-5s).
  5. Extract that audio segment from the raw session .ogg file.
  6. Save extracted segment as WAV to output audios dir.
  7. Write pool JSON.

VoxPopuli layout:
  raw_audios/original/[year]/[session_id]_original.ogg   (ASR-subset download path)
  raw_audios/en/[year]/[session_id]_en.ogg               (unlabelled-subset download path, fallback)

Annotation TSV columns (pipe-delimited):
  session_id | id_ | vad | split | original_text | normed_text |
  speaker_id | gender | is_gold_transcript | accent

Output JSON format (matches LibriSpeech / Common Voice pool):
  {
    "id": "<speaker_id>",
    "ref_path": "audios/<speaker_id>.wav",
    "ref_text": "transcript",
    "original_path": "raw_audios/en/2019/20190107-..._en.ogg",
    "duration": 4.12,
    "gender": "male" | "female" | null,
    "accent": null,
    "age_range": null,
    "energy": null
  }
"""

import argparse
import csv
import gzip
import json
import logging
import multiprocessing as mp
import subprocess
import sys
import time
import tempfile
from ast import literal_eval
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Hardcoded paths ──────────────────────────────────────────────────────────
VOXPOPULI_ROOT = Path(
    "/fs/nexus-projects/brain_project/ashish/speech_data/voxpopuli/voxpopuli"
)
OUTPUT_DIR = Path(
    "/fs/gamma-projects/audio/raman/steerd/steerduplex/src/data/voices/user"
)
AUDIOS_SUBDIR = "voxpopuli_audios"

# ── Annotation TSV URL ────────────────────────────────────────────────────────
ANNOTATION_URL = "https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_en.tsv.gz"

# ── Defaults ─────────────────────────────────────────────────────────────────
TARGET_DUR = 4.0     # seconds
TOLERANCE = 1.5      # accept clips in [target - tol, target + tol]  → 2.5s – 5.5s
DEFAULT_WORKERS = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m"


def normalize_gender(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.strip().lower()
    if r in ("male", "masculine"):
        return "male"
    if r in ("female", "feminine"):
        return "female"
    return None


def vad_duration(vad_str: str) -> float:
    """Compute total duration (seconds) from a VAD string like '[(0.5, 3.2), (3.5, 5.0)]'."""
    try:
        timestamps = literal_eval(vad_str)
        return sum(float(e) - float(s) for s, e in timestamps)
    except Exception:
        return 0.0


def find_session_audio(vox_root: Path, session_id: str) -> Path | None:
    """
    Locate the raw session .ogg for a given session_id.
    Tries ASR-subset path first, then unlabelled-subset path.
    """
    year = session_id[:4]
    candidates = [
        vox_root / "raw_audios" / "original" / year / f"{session_id}_original.ogg",
        vox_root / "raw_audios" / "en" / year / f"{session_id}_en.ogg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── Step 1: Download / cache annotation TSV ──────────────────────────────────

def get_annotation_tsv(cache_dir: Path) -> Path:
    """Download asr_en.tsv.gz to cache_dir if not already there. Returns path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / "asr_en.tsv.gz"
    if gz_path.exists():
        log.info("Using cached annotation file: %s", gz_path)
        return gz_path
    log.info("Downloading annotation TSV from %s ...", ANNOTATION_URL)
    t0 = time.time()
    result = subprocess.run(
        ["wget", "-q", "-O", str(gz_path), ANNOTATION_URL],
        capture_output=True,
    )
    if result.returncode != 0 or not gz_path.exists():
        log.error("Download failed: %s", result.stderr.decode())
        sys.exit(1)
    log.info("  Downloaded in %.1fs  (%s)", time.time() - t0, gz_path)
    return gz_path


# ── Step 2: Parse TSV, group by speaker ──────────────────────────────────────

def load_and_group_train(gz_path: Path, vox_root: Path) -> dict[str, list[dict]]:
    """
    Parse asr_en.tsv.gz, keep only train split, group by speaker_id.
    Skips utterances whose session audio file doesn't exist on disk.
    Returns {speaker_id: [list of utterance dicts]}.
    """
    log.info("Parsing annotation TSV ...")
    t0 = time.time()

    speakers: dict[str, list[dict]] = defaultdict(list)
    total = 0
    skipped_split = 0
    skipped_audio = 0

    # Cache of session_id → audio path to avoid repeated filesystem lookups
    session_cache: dict[str, Path | None] = {}

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            total += 1
            if row.get("split", "").strip() != "train":
                skipped_split += 1
                continue

            session_id = row.get("session_id", "").strip()
            if session_id not in session_cache:
                session_cache[session_id] = find_session_audio(vox_root, session_id)
            audio_path = session_cache[session_id]
            if audio_path is None:
                skipped_audio += 1
                continue

            vad_str = row.get("vad", "").strip()
            dur = vad_duration(vad_str)

            speakers[row["speaker_id"].strip()].append({
                "session_id": session_id,
                "utt_id": row.get("id_", "").strip(),
                "vad": vad_str,
                "duration": dur,
                "text": row.get("normed_text", "").strip() or row.get("original_text", "").strip(),
                "gender": row.get("gender", "").strip() or None,
                "accent": row.get("accent", "").strip() or None,
                "audio_path": audio_path,
            })

    elapsed = time.time() - t0
    log.info(
        "  Parsed %d rows in %.1fs | train: %d | skipped split: %d | skipped (no audio): %d | speakers: %d",
        total, elapsed,
        total - skipped_split - skipped_audio,
        skipped_split, skipped_audio, len(speakers),
    )
    return dict(speakers)


# ── Step 3: Pick best utterance per speaker ───────────────────────────────────

def pick_best_utterance(
    utts: list[dict],
    target_dur: float,
    tolerance: float,
) -> dict | None:
    """Pick utterance closest to target_dur. Prefers text + duration in range."""
    with_text = [u for u in utts if u["text"] and u["duration"] > 0]
    candidates = with_text if with_text else [u for u in utts if u["duration"] > 0]
    if not candidates:
        return utts[0] if utts else None

    # Prefer within [target-tol, target+tol], fall back to closest overall
    in_range = [c for c in candidates if abs(c["duration"] - target_dur) <= tolerance]
    pool = in_range if in_range else candidates
    pool.sort(key=lambda u: abs(u["duration"] - target_dur))
    return pool[0]


# ── Step 4: Extract segment + convert to WAV ─────────────────────────────────

def extract_segment(audio_path: Path, vad_str: str, wav_dest: Path) -> bool:
    """
    Extract VAD-defined segment(s) from session audio, concatenate, save as 16kHz mono WAV.
    Uses soundfile (libsndfile) which natively supports OGG/Vorbis.
    Returns True on success.
    """
    try:
        timestamps = literal_eval(vad_str)
        data, sr = sf.read(str(audio_path), always_2d=False)
        # data shape: (samples,) for mono or (samples, channels) for multi-channel
        if data.ndim > 1:
            data = data[:, 0]  # take first channel
        total_samples = len(data)
        chunks = []
        for s, e in timestamps:
            s_idx = int(float(s) * sr)
            e_idx = min(int(float(e) * sr), total_samples)
            if e_idx > s_idx:
                chunks.append(data[s_idx:e_idx])
        if not chunks:
            return False
        segment = np.concatenate(chunks)
        # Resample to 16kHz if needed via ffmpeg (avoid scipy dependency)
        if sr != 16000:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, segment, sr)
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1",
                 "-sample_fmt", "s16", str(wav_dest)],
                capture_output=True, timeout=30,
            )
            Path(tmp_path).unlink(missing_ok=True)
        else:
            sf.write(str(wav_dest), segment, sr, subtype="PCM_16")
        return wav_dest.exists()
    except Exception as e:
        log.debug("extract_segment failed for %s: %s", audio_path.name, e)
        return False


# ── Step 5: Worker per speaker ────────────────────────────────────────────────

def process_one_speaker(
    args_tuple: tuple[str, list[dict]],
    audios_dir: Path,
    target_dur: float,
    tolerance: float,
) -> dict | None:
    spk_id, utts = args_tuple

    best = pick_best_utterance(utts, target_dur, tolerance)
    if best is None:
        return None

    wav_name = f"{spk_id}.wav"
    wav_dest = audios_dir / wav_name

    if not wav_dest.exists():
        ok = extract_segment(best["audio_path"], best["vad"], wav_dest)
        if not ok:
            return None

    # Compute actual saved duration
    try:
        info = sf.info(str(wav_dest))
        actual_dur = info.duration
    except Exception:
        actual_dur = best["duration"]

    return {
        "id": spk_id,
        "ref_path": f"{AUDIOS_SUBDIR}/{wav_name}",
        "ref_text": best["text"] if best["text"] else None,
        "original_path": str(best["audio_path"]),
        "duration": round(actual_dur, 2),
        "gender": normalize_gender(best["gender"]),
        "accent": best["accent"] if best["accent"] else None,
        "age_range": None,
        "energy": None,
    }


# ── Step 6: Build pool (multiprocessing) ─────────────────────────────────────

def build_pool(
    speakers: dict[str, list[dict]],
    audios_dir: Path,
    target_dur: float,
    tolerance: float,
    num_workers: int,
) -> list[dict]:
    audios_dir.mkdir(parents=True, exist_ok=True)
    sorted_speakers = sorted(speakers.items())
    total = len(sorted_speakers)

    log.info("Extracting audio for %d speakers with %d workers ...", total, num_workers)
    log.info("-" * 60)

    worker_fn = partial(
        process_one_speaker,
        audios_dir=audios_dir,
        target_dur=target_dur,
        tolerance=tolerance,
    )

    pool_entries = []
    skipped = 0
    completed = 0
    t_start = time.time()

    with mp.Pool(processes=num_workers) as proc_pool:
        for result in proc_pool.imap_unordered(worker_fn, sorted_speakers, chunksize=16):
            completed += 1
            if result is not None:
                pool_entries.append(result)
            else:
                skipped += 1

            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0
                # Gender breakdown so far
                male = sum(1 for e in pool_entries if e["gender"] == "male")
                female = sum(1 for e in pool_entries if e["gender"] == "female")
                log.info(
                    "  [%5d / %d]  %5.1f%%  |  %.1f spk/s  |  elapsed %s  |  ETA %s  "
                    "|  collected: %d (M:%d F:%d)  skipped: %d",
                    completed, total, 100.0 * completed / total,
                    rate, fmt_eta(elapsed), fmt_eta(remaining),
                    len(pool_entries), male, female, skipped,
                )

    elapsed_total = time.time() - t_start
    log.info("-" * 60)
    log.info("Done in %s  |  %d speakers in pool, %d skipped", fmt_eta(elapsed_total), len(pool_entries), skipped)
    return pool_entries


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build VoxPopuli voice pool JSON (train split, unique speakers)."
    )
    parser.add_argument(
        "--vox_root", type=str, default=str(VOXPOPULI_ROOT),
        help="VoxPopuli root directory containing raw_audios/ (default: hardcoded path)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for JSON + audios/ (default: hardcoded path)",
    )
    parser.add_argument(
        "--target_dur", type=float, default=TARGET_DUR,
        help="Target utterance duration in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=TOLERANCE,
        help="Acceptable deviation from target duration (default: 1.5)",
    )
    parser.add_argument(
        "--max_speakers", type=int, default=None,
        help="Limit number of speakers (for testing). Default: all.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel workers for audio extraction (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory to cache downloaded annotation TSV (default: <vox_root>/annotations)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    vox_root = Path(args.vox_root).resolve()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else vox_root / "annotations"
    audios_dir = output_dir / AUDIOS_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("VoxPopuli Pool Builder")
    log.info("=" * 60)
    log.info("VoxPopuli root : %s", vox_root)
    log.info("Output dir     : %s", output_dir)
    log.info("Audios dir     : %s", audios_dir)
    log.info("Target dur     : %.1fs ± %.1fs", args.target_dur, args.tolerance)
    log.info("Workers        : %d", args.workers)
    if args.max_speakers:
        log.info("Max speakers   : %d (test mode)", args.max_speakers)
    log.info("=" * 60)

    # ── Step 1: Get annotation TSV ────────────────────────────────────────
    gz_path = get_annotation_tsv(cache_dir)

    # ── Step 2: Parse + group by speaker ─────────────────────────────────
    speakers = load_and_group_train(gz_path, vox_root)

    if not speakers:
        log.error("No speakers found — check that raw session audio files exist in %s", vox_root / "raw_audios")
        sys.exit(1)

    if args.max_speakers:
        speakers = dict(list(speakers.items())[:args.max_speakers])
        log.info("Trimmed to %d speakers for testing", len(speakers))

    # ── Summary ───────────────────────────────────────────────────────────
    total_utts = sum(len(v) for v in speakers.values())
    log.info("Speakers: %d  |  Total train utterances: %d  |  Avg utts/speaker: %.1f",
             len(speakers), total_utts, total_utts / max(len(speakers), 1))

    gender_counts: dict[str, int] = defaultdict(int)
    for utts in speakers.values():
        g = normalize_gender(utts[0].get("gender"))
        gender_counts[str(g)] += 1
    log.info("Gender distribution: %s",
             ", ".join(f"{k}: {v}" for k, v in sorted(gender_counts.items())))

    # Filter to male/female only
    before = len(speakers)
    speakers = {
        spk_id: utts for spk_id, utts in speakers.items()
        if normalize_gender(utts[0].get("gender")) in ("male", "female")
    }
    log.info("Filtered to male/female: %d -> %d (dropped %d with unknown gender)",
             before, len(speakers), before - len(speakers))

    # ── Step 3–5: Build pool ──────────────────────────────────────────────
    pool = build_pool(speakers, audios_dir, args.target_dur, args.tolerance, args.workers)

    # ── Step 6: Save JSON ─────────────────────────────────────────────────
    json_path = output_dir / "voxpopuli_voice_pool.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    log.info("Saved pool JSON: %s", json_path)

    # ── Final summary ─────────────────────────────────────────────────────
    if pool:
        durations = [e["duration"] for e in pool]
        genders: dict[str, int] = defaultdict(int)
        for e in pool:
            genders[str(e["gender"])] += 1
        log.info("=" * 60)
        log.info("FINAL POOL SUMMARY")
        log.info("=" * 60)
        log.info("Total voices   : %d", len(pool))
        log.info("Duration range : %.2fs – %.2fs (mean %.2fs)",
                 min(durations), max(durations), sum(durations) / len(durations))
        log.info("Gender         : %s",
                 ", ".join(f"{k}: {v}" for k, v in sorted(genders.items())))
        log.info("JSON saved to  : %s", json_path)
        log.info("Audios in      : %s", audios_dir)
    log.info("Done!")


if __name__ == "__main__":
    main()
