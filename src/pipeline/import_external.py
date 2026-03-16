"""Import external datasets into standard SteerDuplex format.

Standard format for ALL datasets (synthetic + external):
    data/external/{name}/
    ├── manifest.jsonl       # {"path": "audio/xxx.wav", "duration": 45.2}
    └── audio/
        ├── xxx.wav          # stereo WAV (ch0=assistant, ch1=user)
        └── xxx.json         # word-level alignments (moshi format)

This script is a ONE-TIME prep step per dataset. It converts raw data into
the standard format. After import, just add the dataset to full_training.yaml.

Usage:
    python -m pipeline.import_external --dataset annutacon
    python -m pipeline.import_external --dataset fisher
    python -m pipeline.import_external --list
"""

import argparse
import json
import os
import tarfile
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from pipeline.utils import ensure_dir


def get_audio_duration(path: Path) -> float:
    try:
        return sf.info(str(path)).duration
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Standard format for each dataset:
#   data/external/{name}/manifest.jsonl
#   data/external/{name}/audio/{prefix}_{id}.wav  (symlinked to source)
#   data/external/{name}/audio/{prefix}_{id}.json (symlinked to source)
# ---------------------------------------------------------------------------

def _symlink(src: Path, dst: Path):
    """Create symlink if it doesn't exist."""
    if not dst.exists() and not dst.is_symlink():
        os.symlink(str(src.resolve()), str(dst))


def import_annutacon(base_dir: Path) -> int:
    """Import annutacon into standard format.

    Raw structure:
        data/external/annutacon/data_stereo/{id}.wav
        data/external/annutacon/data_stereo/{id}.json (moshi alignments)
        data/external/annutacon/dataset.jsonl
    """
    src_audio = base_dir / "data_stereo"
    if not src_audio.exists():
        print(f"[ERROR] {src_audio} not found")
        return 0

    audio_dir = ensure_dir(base_dir / "audio")
    manifest_path = base_dir / "manifest.jsonl"

    # Load durations from dataset.jsonl if available
    durations = {}
    dataset_jsonl = base_dir / "dataset.jsonl"
    if dataset_jsonl.exists():
        with open(dataset_jsonl) as f:
            for line in f:
                d = json.loads(line)
                stem = Path(d["path"]).stem
                durations[stem] = d["duration"]

    entries = []
    for wav_path in tqdm(sorted(src_audio.glob("*.wav")), desc="annutacon"):
        stem = wav_path.stem
        json_path = src_audio / f"{stem}.json"

        dst_wav = audio_dir / f"annutacon_{stem}.wav"
        dst_json = audio_dir / f"annutacon_{stem}.json"

        _symlink(wav_path, dst_wav)
        if json_path.exists():
            _symlink(json_path, dst_json)

        duration = durations.get(stem) or get_audio_duration(wav_path)
        entries.append({"path": f"audio/annutacon_{stem}.wav", "duration": round(duration, 2)})

    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"annutacon: {len(entries)} files, {total_hours:.0f}h → {manifest_path}")
    return len(entries)


def import_fisher(base_dir: Path) -> int:
    """Import Fisher into standard format.

    Raw structure (nested):
        data/external/fisher/fisher_wav.tar (or already extracted)
        data/external/fisher/fisher_wav/fe_03_*/audio/{subdir}/fe_03_{id}.wav
        data/external/fisher/fisher_wav/fe_03_*/audio/{subdir}/fe_03_{id}.json
    """
    tar_path = base_dir / "fisher_wav.tar"
    wav_root = base_dir / "fisher_wav"

    if tar_path.exists() and (not wav_root.exists() or len(list(wav_root.rglob("*.wav"))) < 10):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(base_dir, filter="data")

    if not wav_root.exists():
        print(f"[ERROR] {wav_root} not found")
        return 0

    audio_dir = ensure_dir(base_dir / "audio")
    manifest_path = base_dir / "manifest.jsonl"

    wav_files = sorted(wav_root.rglob("*.wav"))
    print(f"Fisher: found {len(wav_files)} WAVs")

    entries = []
    for wav_path in tqdm(wav_files, desc="fisher"):
        stem = wav_path.stem
        json_path = wav_path.with_suffix(".json")
        duration = get_audio_duration(wav_path)

        if duration < 10:
            continue

        dst_wav = audio_dir / f"fisher_{stem}.wav"
        dst_json = audio_dir / f"fisher_{stem}.json"

        _symlink(wav_path, dst_wav)
        if json_path.exists():
            _symlink(json_path, dst_json)

        entries.append({"path": f"audio/fisher_{stem}.wav", "duration": round(duration, 2)})

    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"fisher: {len(entries)} files, {total_hours:.0f}h → {manifest_path}")
    return len(entries)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
IMPORTERS = {
    "annutacon": ("data/external/annutacon", import_annutacon),
    "fisher": ("data/external/fisher", import_fisher),
}


def main():
    parser = argparse.ArgumentParser(description="Import external datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--list", action="store_true", help="Show dataset status")
    args = parser.parse_args()

    if args.list:
        print(f"{'Dataset':15s} {'Dir exists':12s} {'Manifest':12s} {'Hours':>8s}")
        print("-" * 50)
        for name, (default_dir, _) in IMPORTERS.items():
            d = Path(default_dir)
            exists = "YES" if d.exists() else "NO"
            manifest = d / "manifest.jsonl"
            if manifest.exists():
                hours = 0
                with open(manifest) as f:
                    for line in f:
                        hours += json.loads(line).get("duration", 0)
                hours_str = f"{hours/3600:.0f}h"
                manifest_str = "YES"
            else:
                hours_str = "-"
                manifest_str = "NO"
            print(f"{name:15s} {exists:12s} {manifest_str:12s} {hours_str:>8s}")
        return

    if not args.dataset:
        parser.error("--dataset required (or use --list)")

    if args.dataset not in IMPORTERS:
        print(f"Unknown: {args.dataset}. Available: {list(IMPORTERS.keys())}")
        return

    default_dir, importer = IMPORTERS[args.dataset]
    importer(Path(default_dir))


if __name__ == "__main__":
    main()
