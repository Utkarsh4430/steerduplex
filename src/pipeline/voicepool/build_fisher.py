"""
Build a voice pool JSON from the Fisher English Part 2 dataset.

Strategy (efficient — speaker-first):
  1. Load filelist.tbl  → {call_id: sph_dir}  (set of available Part 2 calls)
  2. Load pindata.tbl   → 11k unique speakers with gender, age, side_data
  3. Pre-index all transcripts for available calls → {(call_id, channel): [segments]}
     Only keeps segments within [MIN_DUR, MAX_DUR] — avoids loading same file twice.
  4. For each speaker, look up their call appearances in the index, pick best segment.
  5. Parallel ffmpeg extractions: stereo WAV → mono channel WAV for the chosen segment.

Fisher layout (Part 2):
  fisher_wav/fe_03_p2_sph[1-7]/audio/[XXX]/fe_03_XXXXX.wav   stereo WAVs
  fisher/fe_03_p2_tran/doc/fe_03_pindata.tbl                  speaker metadata
  fisher/fe_03_p2_tran/doc/fe_03_p2_filelist.tbl              call → sph partition
  fisher/fe_03_p2_tran/data/trans/[XXX]/fe_03_XXXXX.txt       LDC transcripts
  fisher/fe_03_p2_tran/data/bbn_orig/[XXX]/auto-segmented/    BBN transcripts

Fisher stereo layout:
  Left channel  (index 0) = speaker A
  Right channel (index 1) = speaker B

Output JSON format:
  {
    "id": "96498",
    "ref_path": "fisher_audios/96498.wav",
    "ref_text": "transcript text",
    "original_path": "fe_03_p2_sph1/audio/058/fe_03_05851.wav",
    "duration": 4.2,
    "gender": "female",
    "accent": "american",       # from dialect field (.a / .o) or null
    "age_range": "fifties",     # converted from numeric age or null
    "energy": null
  }
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import re
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

# ── Hardcoded paths ───────────────────────────────────────────────────────────
FISHER_WAV_DIR  = Path("/fs/gamma-projects/audio/audio_datasets/fisher_wav")
FISHER_META_DIR = Path("/fs/gamma-projects/audio/audio_datasets/fisher/fe_03_p2_tran")
OUTPUT_DIR      = Path("/fs/gamma-projects/audio/raman/steerd/steerduplex/src/data/voices/user")
AUDIOS_SUBDIR   = "fisher_audios"

# ── Duration constraints ──────────────────────────────────────────────────────
MIN_DUR    = 3.0   # seconds
MAX_DUR    = 5.0   # seconds
TARGET_DUR = 4.0   # prefer segments closest to this

# BBN telephone audio sample rate for segment time conversion
FISHER_SAMPLE_RATE = 8000

DEFAULT_WORKERS = 8

# ── Metadata paths ────────────────────────────────────────────────────────────
PINDATA_TBL  = FISHER_META_DIR / "doc" / "fe_03_pindata.tbl"
FILELIST_TBL = FISHER_META_DIR / "doc" / "fe_03_p2_filelist.tbl"
TRANS_DIR    = FISHER_META_DIR / "data" / "trans"
BBN_DIR      = FISHER_META_DIR / "data" / "bbn_orig"


# ── Utilities ─────────────────────────────────────────────────────────────────

def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m = rem // 60
        return f"{int(h)}h {int(m)}m"


def normalize_gender(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.strip().lower()
    if r in ("m", "male"):
        return "male"
    if r in ("f", "female"):
        return "female"
    return None


def dialect_to_accent(dialect_char: str | None) -> str | None:
    """Convert Fisher dialect character ('a'=American, 'o'=Other) to accent label."""
    if not dialect_char:
        return None
    d = dialect_char.strip().lower()
    if d == "a":
        return "american"
    if d == "o":
        return "other"
    return None


def age_to_range(age: int | None) -> str | None:
    """Convert numeric age to decade label matching Common Voice convention."""
    if age is None:
        return None
    if age < 13:
        return None
    if age < 20:
        return "teens"
    if age < 30:
        return "twenties"
    if age < 40:
        return "thirties"
    if age < 50:
        return "forties"
    if age < 60:
        return "fifties"
    if age < 70:
        return "sixties"
    if age < 80:
        return "seventies"
    if age < 90:
        return "eighties"
    return "nineties"


def clean_bbn_text(text: str) -> str:
    """Lowercase BBN uppercase text (no marker stripping — markers cause rejection)."""
    return re.sub(r'\s+', ' ', text).strip().lower()


# ── Step 1: Load filelist ─────────────────────────────────────────────────────

def load_filelist(path: Path) -> dict[str, str]:
    """
    Load fe_03_p2_filelist.tbl → {call_id: sph_dir_name}.
    call_id is a zero-padded 5-digit string (e.g. "05851").
    sph_dir_name is e.g. "fe_03_p2_sph1".
    """
    log.info("Loading filelist: %s", path)
    filelist: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sph_dir  = parts[0]   # e.g. fe_03_p2_sph1
            call_id  = parts[1].zfill(5)
            filelist[call_id] = sph_dir
    log.info("  %d calls mapped to sph partitions", len(filelist))
    return filelist


def wav_path_for_call(call_id: str, sph_dir: str) -> Path:
    """Build absolute WAV path from call_id and sph directory name."""
    subdir = call_id[:3]  # first 3 digits of 5-digit call_id
    return FISHER_WAV_DIR / sph_dir / "audio" / subdir / f"fe_03_{call_id}.wav"


# ── Transcript cleanliness filter ────────────────────────────────────────────

# Reject segments containing any bracketed annotation ([noise], [laughter], etc.)
# or double-paren unclear speech markers (( ))
_DIRTY_PAT = re.compile(r'\[.*?\]|\(\(')

def is_clean_text(text: str) -> bool:
    """Return True only if text has no noise/annotation markers."""
    return bool(text) and not _DIRTY_PAT.search(text)


# ── Step 2: Load transcripts (LDC and BBN) ────────────────────────────────────

# A segment tuple: (start_sec, end_sec, channel_str, text_str)
Segment = tuple[float, float, str, str]


def load_ldc_transcript(call_id: str) -> list[Segment]:
    """
    Parse LDC transcript for call_id.
    Format per line: START END CHANNEL: text
    Returns list of (start, end, channel, text).
    """
    subdir = call_id[:3]
    path = TRANS_DIR / subdir / f"fe_03_{call_id}.txt"
    if not path.exists():
        return []

    segments: list[Segment] = []
    # Pattern: float float [AB]: text
    pat = re.compile(r'^(\d+\.?\d*)\s+(\d+\.?\d*)\s+([AB]):\s*(.*)')
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            start = float(m.group(1))
            end   = float(m.group(2))
            ch    = m.group(3)
            text  = m.group(4).strip()
            segments.append((start, end, ch, text))
    return segments


def load_bbn_transcript(call_id: str) -> list[Segment]:
    """
    Parse BBN transcript for call_id.
    Uses .ana (sample boundaries) + .trn (text) files in bbn_orig/.
    Returns list of (start_sec, end_sec, channel, text).
    Channel derived from -c flag: 1→A, 2→B.
    """
    subdir = call_id[:3]
    search_dir = BBN_DIR / subdir / "auto-segmented"
    if not search_dir.exists():
        return []

    prefix = f"fe_03_{call_id}"
    ana_files = list(search_dir.glob(f"{prefix}*.ana"))
    trn_files = list(search_dir.glob(f"{prefix}*.trn"))
    if not ana_files or not trn_files:
        return []

    # Build utterance_id → text from .trn files
    utt_text: dict[str, str] = {}
    for trn_path in trn_files:
        with open(trn_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: TEXT (utterance_id)
                m = re.match(r'^(.*)\((\S+)\)\s*$', line)
                if m:
                    raw_text = m.group(1).strip()
                    utt_id   = m.group(2)
                    utt_text[utt_id] = clean_bbn_text(raw_text)

    # Parse .ana files: sample boundaries → time
    segments: list[Segment] = []
    ch_pat = re.compile(r'-c\s+(\d+)')
    f_pat  = re.compile(r'-f\s+(\d+)-(\d+)')
    o_pat  = re.compile(r'-o\s+(\S+)')

    for ana_path in ana_files:
        with open(ana_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ch_m = ch_pat.search(line)
                f_m  = f_pat.search(line)
                o_m  = o_pat.search(line)
                if not (ch_m and f_m and o_m):
                    continue
                ch_idx  = int(ch_m.group(1))
                channel = "A" if ch_idx == 1 else "B"
                start_s = int(f_m.group(1)) / FISHER_SAMPLE_RATE
                end_s   = int(f_m.group(2)) / FISHER_SAMPLE_RATE
                utt_id  = o_m.group(1)
                text    = utt_text.get(utt_id, "")
                segments.append((start_s, end_s, channel, text))

    return segments


def load_transcript(call_id: str) -> list[Segment]:
    """Try LDC transcript first, then BBN."""
    segs = load_ldc_transcript(call_id)
    if segs:
        return segs
    return load_bbn_transcript(call_id)


# ── Step 3: Build segment index ───────────────────────────────────────────────

def build_segment_index(
    filelist: dict[str, str],
    min_dur: float,
    max_dur: float,
) -> dict[tuple[str, str], list[Segment]]:
    """
    Load all available transcripts and pre-filter to segments in [min_dur, max_dur].
    Returns {(call_id, channel): [valid segments]} for calls that have any valid segments.
    Logs progress with ETA.
    """
    call_ids = sorted(filelist.keys())
    total    = len(call_ids)
    index: dict[tuple[str, str], list[Segment]] = defaultdict(list)

    log.info("Building segment index from %d calls (transcripts) ...", total)
    t_start = time.time()
    ldc_count = bbn_count = no_trans = no_segs = 0

    for i, call_id in enumerate(call_ids, 1):
        ldc_segs = load_ldc_transcript(call_id)
        if ldc_segs:
            segs = ldc_segs
            ldc_count += 1
        else:
            segs = load_bbn_transcript(call_id)
            if segs:
                bbn_count += 1
            else:
                no_trans += 1

        # Filter to duration range and clean text only
        valid = [(s, e, ch, t) for (s, e, ch, t) in segs
                 if min_dur <= (e - s) <= max_dur and is_clean_text(t)]
        if not valid:
            no_segs += 1
        else:
            for seg in valid:
                index[(call_id, seg[2])].append(seg)

        # Progress every 500 calls
        if i % 500 == 0 or i == total:
            elapsed   = time.time() - t_start
            rate      = i / elapsed
            remaining = (total - i) / rate if rate > 0 else 0
            log.info(
                "  [%5d / %d]  %5.1f%%  |  %.0f calls/s  |  elapsed %s  |  ETA %s"
                "  |  LDC: %d  BBN: %d  no-trans: %d",
                i, total, 100.0 * i / total, rate,
                fmt_eta(elapsed), fmt_eta(remaining),
                ldc_count, bbn_count, no_trans,
            )

    indexed_calls = len({k[0] for k in index})
    log.info(
        "Segment index built: %d call-channel pairs across %d calls  |  "
        "no-segments: %d calls",
        len(index), indexed_calls, no_segs,
    )
    return dict(index)


# ── Step 4: Load speaker metadata ─────────────────────────────────────────────

def parse_side_data(side_data: str, available_calls: set[str]) -> list[tuple[str, str, str | None]]:
    """
    Parse pindata.tbl SIDE_DATA field.
    Format: CALLID_CHANNEL/gender.dialect;...  (e.g. "05851_A/f.a;06411_B/m.a")
    Returns list of (call_id, channel, dialect_char) for calls in available_calls.
    """
    results = []
    for entry in side_data.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        # e.g. "05851_A/f.a" or "05851_A" (no gender part)
        m = re.match(r'^(\d+)_([AB])(?:/([mf])\.([ao]))?', entry)
        if not m:
            continue
        call_id  = m.group(1).zfill(5)
        channel  = m.group(2)
        dialect  = m.group(4)  # 'a' or 'o', may be None
        if call_id in available_calls:
            results.append((call_id, channel, dialect))
    return results


def load_pindata(
    path: Path,
    available_calls: set[str],
) -> dict[str, dict]:
    """
    Load fe_03_pindata.tbl → {pin: speaker_dict}.
    speaker_dict keys: gender, age, age_range, native_lang, where_raised,
                       sides [(call_id, channel, dialect), ...]
    Only includes speakers with at least one appearance in available_calls.
    """
    log.info("Loading pindata: %s", path)
    t0 = time.time()
    speakers: dict[str, dict] = {}
    total_rows = 0
    dropped_no_call = 0
    dropped_no_gender = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            pin       = row.get("PIN", "").strip()
            sex_raw   = row.get("S_SEX", "").strip()
            age_raw   = row.get("S_AGE", "").strip()
            edu       = row.get("EDU", "").strip()
            nat_lang  = row.get("NATIVE_LANG", "").strip()
            raised    = row.get("WHERE_RAISED", "").strip()
            side_data = row.get("SIDE_DATA", "").strip()

            if not pin:
                continue

            gender = normalize_gender(sex_raw)
            if gender is None:
                dropped_no_gender += 1
                continue

            try:
                age_int = int(age_raw)
            except ValueError:
                age_int = None

            sides = parse_side_data(side_data, available_calls)
            if not sides:
                dropped_no_call += 1
                continue

            speakers[pin] = {
                "pin":         pin,
                "gender":      gender,
                "age":         age_int,
                "age_range":   age_to_range(age_int),
                "native_lang": nat_lang if nat_lang else None,
                "where_raised": raised if raised else None,
                "sides":       sides,   # [(call_id, channel, dialect), ...]
            }

    elapsed = time.time() - t0
    log.info(
        "  Parsed %d speakers in %.1fs  |  valid: %d  "
        "|  dropped no-gender: %d  dropped no-call: %d",
        total_rows, elapsed, len(speakers), dropped_no_gender, dropped_no_call,
    )
    return speakers


# ── Step 5: Pick best segment per speaker ─────────────────────────────────────

def pick_best_segment(
    sides: list[tuple[str, str, str | None]],
    segment_index: dict[tuple[str, str], list[Segment]],
    target_dur: float,
) -> tuple[Segment, str, str] | None:
    """
    For a speaker's list of (call_id, channel, dialect) appearances, find the
    segment closest to target_dur from the pre-built index.
    Returns (segment, call_id, channel) or None.
    """
    best_seg  = None
    best_dist = float("inf")
    best_call = None
    best_chan = None

    for call_id, channel, _ in sides:
        segs = segment_index.get((call_id, channel), [])
        for seg in segs:
            dur  = seg[1] - seg[0]
            dist = abs(dur - target_dur)
            if dist < best_dist:
                best_dist = dist
                best_seg  = seg
                best_call = call_id
                best_chan = channel

    if best_seg is None:
        return None
    return (best_seg, best_call, best_chan)


# ── Step 6: Audio extraction ──────────────────────────────────────────────────

def extract_segment(
    src_wav: Path,
    channel: str,       # "A" or "B"
    start: float,
    end: float,
    dst_wav: Path,
) -> bool:
    """
    Extract mono segment from stereo Fisher WAV via ffmpeg.
    Channel A → left (index 0), Channel B → right (index 1).
    Output: 16kHz mono 16-bit PCM WAV.
    """
    # FL = front-left = channel A, FR = front-right = channel B
    pan_ch = "FL" if channel == "A" else "FR"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(src_wav),
                "-af", f"pan=mono|c0={pan_ch}",
                "-ar", "16000",
                "-sample_fmt", "s16",
                str(dst_wav),
            ],
            capture_output=True,
            timeout=30,
        )
        if not (dst_wav.exists() and dst_wav.stat().st_size > 0):
            log.debug("ffmpeg stderr: %s", result.stderr.decode(errors="replace")[-300:])
            return False
        return True
    except Exception as e:
        log.warning("ffmpeg failed for %s ch%s [%.2f-%.2f]: %s", src_wav.name, channel, start, end, e)
        return False


def get_actual_duration(wav_path: Path) -> float | None:
    """Probe actual duration of extracted WAV via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(wav_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# ── Worker function ───────────────────────────────────────────────────────────

def process_one_speaker(
    spk: dict,
    filelist: dict[str, str],
    segment_index: dict[tuple[str, str], list[Segment]],
    audios_dir: Path,
    target_dur: float,
) -> dict | None:
    """
    Worker: find best segment for one speaker, extract WAV, return pool entry.
    """
    pin   = spk["pin"]
    sides = spk["sides"]

    result = pick_best_segment(sides, segment_index, target_dur)
    if result is None:
        return None

    seg, call_id, channel = result
    start, end, _, text = seg

    # Build source WAV path
    sph_dir = filelist.get(call_id)
    if sph_dir is None:
        return None
    src_wav = wav_path_for_call(call_id, sph_dir)
    if not src_wav.exists():
        log.warning("WAV not found: %s", src_wav)
        return None

    # Destination WAV
    wav_name = f"{pin}.wav"
    dst_wav  = audios_dir / wav_name

    if not dst_wav.exists():
        ok = extract_segment(src_wav, channel, start, end, dst_wav)
        if not ok:
            log.warning("Extraction failed: %s ch%s [%.2f-%.2f]", src_wav.name, channel, start, end)
            return None

    # Probe actual duration of extracted file
    actual_dur = get_actual_duration(dst_wav) or round(end - start, 2)

    # Derive accent from first available dialect char in sides
    accent = None
    for _, ch, dialect in sides:
        if ch == channel and dialect is not None:
            accent = dialect_to_accent(dialect)
            break

    # original_path relative to fisher_wav root
    subdir = call_id[:3]
    original_path = f"{sph_dir}/audio/{subdir}/fe_03_{call_id}.wav"

    return {
        "id":            pin,
        "ref_path":      f"{AUDIOS_SUBDIR}/{wav_name}",
        "ref_text":      text if text else None,
        "original_path": original_path,
        "duration":      round(actual_dur, 2),
        "gender":        spk["gender"],
        "accent":        accent,
        "age_range":     spk["age_range"],
        "energy":        None,
    }


# ── Pool builder with progress + ETA ─────────────────────────────────────────

def build_voice_pool(
    speakers: dict[str, dict],
    filelist: dict[str, str],
    segment_index: dict[tuple[str, str], list[Segment]],
    output_dir: Path,
    target_dur: float,
    num_workers: int,
) -> list[dict]:
    audios_dir = output_dir / AUDIOS_SUBDIR
    audios_dir.mkdir(parents=True, exist_ok=True)

    sorted_spks = sorted(speakers.values(), key=lambda s: s["pin"])
    total = len(sorted_spks)

    log.info("Processing %d speakers with %d workers ...", total, num_workers)
    log.info("-" * 70)

    worker_fn = partial(
        process_one_speaker,
        filelist=filelist,
        segment_index=segment_index,
        audios_dir=audios_dir,
        target_dur=target_dur,
    )

    pool_entries = []
    skipped = 0
    t_start = time.time()
    completed = 0

    with mp.Pool(processes=num_workers) as proc_pool:
        for result in proc_pool.imap_unordered(worker_fn, sorted_spks, chunksize=16):
            completed += 1
            if result is not None:
                pool_entries.append(result)
            else:
                skipped += 1

            if completed % 100 == 0 or completed == total:
                elapsed   = time.time() - t_start
                rate      = completed / elapsed
                remaining = (total - completed) / rate if rate > 0 else 0
                pct       = 100.0 * completed / total
                log.info(
                    "  [%5d / %d]  %5.1f%%  |  %.1f spk/s  |  elapsed %s  |  ETA %s"
                    "  |  pool: %d  skipped: %d",
                    completed, total, pct, rate,
                    fmt_eta(elapsed), fmt_eta(remaining),
                    len(pool_entries), skipped,
                )

    elapsed_total = time.time() - t_start
    log.info("-" * 70)
    log.info(
        "Done in %s  |  %d speakers in pool, %d skipped",
        fmt_eta(elapsed_total), len(pool_entries), skipped,
    )
    return pool_entries


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Fisher Part 2 voice pool JSON (efficient, speaker-first)."
    )
    parser.add_argument("--output_dir",   type=str,   default=str(OUTPUT_DIR))
    parser.add_argument("--target_dur",   type=float, default=TARGET_DUR,
                        help="Target segment duration in seconds (default: 4.0)")
    parser.add_argument("--min_dur",      type=float, default=MIN_DUR,
                        help="Minimum segment duration (default: 3.0)")
    parser.add_argument("--max_dur",      type=float, default=MAX_DUR,
                        help="Maximum segment duration (default: 5.0)")
    parser.add_argument("--workers",      type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--max_speakers", type=int,   default=None,
                        help="Limit speakers for testing (default: all)")
    parser.add_argument("--debug",        action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Fisher Part 2 Voice Pool Builder")
    log.info("=" * 70)
    log.info("Fisher WAV dir  : %s", FISHER_WAV_DIR)
    log.info("Fisher meta dir : %s", FISHER_META_DIR)
    log.info("Output dir      : %s", output_dir)
    log.info("Duration range  : %.1fs – %.1fs  (target: %.1fs)",
             args.min_dur, args.max_dur, args.target_dur)
    log.info("Workers         : %d", args.workers)
    if args.max_speakers:
        log.info("Max speakers    : %d (test mode)", args.max_speakers)
    log.info("=" * 70)

    # ── Step 1: Load filelist ─────────────────────────────────────────────
    filelist      = load_filelist(FILELIST_TBL)
    available_calls = set(filelist.keys())

    # ── Step 2: Load speaker metadata ────────────────────────────────────
    speakers = load_pindata(PINDATA_TBL, available_calls)

    if args.max_speakers:
        speakers = dict(list(speakers.items())[:args.max_speakers])
        log.info("Trimmed to %d speakers for testing", len(speakers))

    # Stats before processing
    gender_counts: dict[str, int] = defaultdict(int)
    age_counts:    dict[str, int] = defaultdict(int)
    for s in speakers.values():
        gender_counts[s["gender"]] += 1
        if s["age_range"]:
            age_counts[s["age_range"]] += 1

    log.info("Speaker count   : %d", len(speakers))
    log.info("Gender dist     : %s",
             "  ".join(f"{k}: {v}" for k, v in sorted(gender_counts.items())))
    log.info("Age dist        : %s",
             "  ".join(f"{k}: {v}" for k, v in sorted(age_counts.items())))

    # ── Step 3: Build segment index ───────────────────────────────────────
    # Only index calls that are actually referenced by our speakers
    needed_calls = {call_id for s in speakers.values() for call_id, _, _ in s["sides"]}
    needed_filelist = {c: filelist[c] for c in needed_calls if c in filelist}
    log.info("Indexing transcripts for %d calls (referenced by speakers) ...",
             len(needed_filelist))
    segment_index = build_segment_index(needed_filelist, args.min_dur, args.max_dur)

    # Drop speakers with no indexed segments
    before = len(speakers)
    speakers = {
        pin: s for pin, s in speakers.items()
        if any((cid, ch) in segment_index for cid, ch, _ in s["sides"])
    }
    log.info("Speakers with valid segments: %d → %d (dropped %d with no segments)",
             before, len(speakers), before - len(speakers))

    # ── Step 4: Build pool (parallel ffmpeg) ──────────────────────────────
    pool = build_voice_pool(
        speakers, filelist, segment_index, output_dir,
        args.target_dur, args.workers,
    )

    # ── Step 5: Save JSON ─────────────────────────────────────────────────
    json_path = output_dir / "fisher_voice_pool.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    log.info("Saved pool JSON: %s", json_path)

    # ── Final summary ─────────────────────────────────────────────────────
    if pool:
        durations = [e["duration"] for e in pool]
        genders   = defaultdict(int)
        accents   = defaultdict(int)
        ages      = defaultdict(int)
        for e in pool:
            genders[str(e["gender"])] += 1
            if e["accent"]:
                accents[e["accent"]] += 1
            if e["age_range"]:
                ages[e["age_range"]] += 1

        log.info("=" * 70)
        log.info("FINAL POOL SUMMARY")
        log.info("=" * 70)
        log.info("Total voices    : %d", len(pool))
        log.info("Duration range  : %.2fs – %.2fs  (mean %.2fs)",
                 min(durations), max(durations), sum(durations) / len(durations))
        log.info("Gender          : %s",
                 "  ".join(f"{k}: {v}" for k, v in sorted(genders.items())))
        log.info("Accent          : %s",
                 "  ".join(f"{k}: {v}" for k, v in sorted(accents.items())))
        log.info("Age range       : %s",
                 "  ".join(f"{k}: {v}" for k, v in sorted(ages.items(), key=lambda x: -x[1])))
        log.info("JSON saved to   : %s", json_path)
        log.info("Audio files in  : %s", output_dir / AUDIOS_SUBDIR)

    log.info("Done!")


if __name__ == "__main__":
    main()
