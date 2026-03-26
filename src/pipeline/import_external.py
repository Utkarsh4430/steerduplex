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

_USE_COPY = False  # Set via --copy CLI flag

# Paths that are local-only (not accessible across nodes via EFS/NFS).
# Files under these prefixes are always copied, never symlinked.
_LOCAL_ONLY_PREFIXES = ("/opt/dlami/nvme", "/mnt/nvme", "/tmp", "/scratch")


def _link_or_copy(src: Path, dst: Path):
    """Create symlink or copy file. Auto-copies if source is on local-only storage."""
    if dst.exists() or dst.is_symlink():
        return
    resolved = str(src.resolve())
    force_copy = _USE_COPY or any(resolved.startswith(p) for p in _LOCAL_ONLY_PREFIXES)
    if force_copy:
        import shutil
        shutil.copy2(str(src), str(dst))
    else:
        os.symlink(resolved, str(dst))


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

        _link_or_copy(wav_path, dst_wav)
        if json_path.exists():
            _link_or_copy(json_path, dst_json)

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

    Supports both old and new directory structures:
        Old: data/external/fisher/fisher_wav/fe_03_*/audio/{subdir}/*.wav
        New: data/external/fisher/fisher_p1_wav/*/audio/{subdir}/*.wav
             data/external/fisher/fisher_p2_wav/*/audio/{subdir}/*.wav
    """
    audio_dir = ensure_dir(base_dir / "audio")
    manifest_path = base_dir / "manifest.jsonl"

    # Find all WAV files across any subdirectory, excluding our output audio/ dir
    wav_files = sorted(base_dir.rglob("*.wav"))
    wav_files = [w for w in wav_files if not str(w.resolve()).startswith(str(audio_dir.resolve()))]
    print(f"Fisher: found {len(wav_files)} source WAVs")

    entries = []
    for wav_path in tqdm(wav_files, desc="fisher"):
        stem = wav_path.stem
        json_path = wav_path.with_suffix(".json")
        duration = get_audio_duration(wav_path)

        if duration < 10:
            continue

        dst_wav = audio_dir / f"fisher_{stem}.wav"
        dst_json = audio_dir / f"fisher_{stem}.json"

        _link_or_copy(wav_path, dst_wav)
        if json_path.exists():
            _link_or_copy(json_path, dst_json)

        entries.append({"path": f"audio/fisher_{stem}.wav", "duration": round(duration, 2)})

    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"fisher: {len(entries)} files, {total_hours:.0f}h → {manifest_path}")
    return len(entries)


def import_generic(base_dir: Path) -> int:
    """Import any stereo WAV dataset into standard format.

    Expected structure:
        data/external/{name}/raw/        # stereo WAVs + optional .json alignments
        OR
        data/external/{name}/data/       # alternative location

    Looks for WAVs in raw/, data/, or the base_dir itself.
    After import, run `pipeline.add_prompts_external --dataset {name}` to add
    system prompt regions with LLM-generated prompts.
    """
    # Find WAVs in common subdirectories
    audio_src = None
    for subdir in ["raw", "data", "data_stereo", "audio_raw", ""]:
        candidate = base_dir / subdir if subdir else base_dir
        wavs = list(candidate.glob("*.wav"))
        if wavs:
            audio_src = candidate
            break

    if audio_src is None:
        print(f"[ERROR] No WAVs found in {base_dir} or its subdirectories (raw/, data/, etc.)")
        return 0

    dataset_name = base_dir.name
    audio_dir = ensure_dir(base_dir / "audio")
    manifest_path = base_dir / "manifest.jsonl"

    wav_files = sorted(audio_src.glob("*.wav"))
    print(f"{dataset_name}: found {len(wav_files)} WAVs in {audio_src}")

    entries = []
    for wav_path in tqdm(wav_files, desc=dataset_name):
        stem = wav_path.stem
        json_path = wav_path.with_suffix(".json")
        duration = get_audio_duration(wav_path)

        if duration < 5:
            continue

        dst_wav = audio_dir / f"{dataset_name}_{stem}.wav"
        dst_json = audio_dir / f"{dataset_name}_{stem}.json"

        _link_or_copy(wav_path, dst_wav)
        if json_path.exists():
            _link_or_copy(json_path, dst_json)

        entries.append({"path": f"audio/{dataset_name}_{stem}.wav", "duration": round(duration, 2)})

    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"{dataset_name}: {len(entries)} files, {total_hours:.0f}h → {manifest_path}")
    print(f"\nNext: run system prompt injection:")
    print(f"  python -m pipeline.add_prompts_external --dataset {dataset_name} --use_llm")
    return len(entries)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def import_annutacon_studio(base_dir: Path) -> int:
    """Import annutacon_studio (studio-quality English SFT conversations).

    Same format as annutacon: data_stereo/{id}.wav + {id}.json
    """
    src_audio = base_dir / "data_stereo"
    if not src_audio.exists():
        print(f"[ERROR] {src_audio} not found")
        return 0

    audio_dir = ensure_dir(base_dir / "audio")
    manifest_path = base_dir / "manifest.jsonl"

    durations = {}
    dataset_jsonl = base_dir / "dataset.jsonl"
    if dataset_jsonl.exists():
        with open(dataset_jsonl) as f:
            for line in f:
                d = json.loads(line)
                stem = Path(d["path"]).stem
                durations[stem] = d["duration"]

    entries = []
    for wav_path in tqdm(sorted(src_audio.glob("*.wav")), desc="annutacon_studio"):
        stem = wav_path.stem
        json_path = src_audio / f"{stem}.json"

        dst_wav = audio_dir / f"annutacon_studio_{stem}.wav"
        dst_json = audio_dir / f"annutacon_studio_{stem}.json"

        _link_or_copy(wav_path, dst_wav)
        if json_path.exists():
            _link_or_copy(json_path, dst_json)

        duration = durations.get(stem) or get_audio_duration(wav_path)
        entries.append({"path": f"audio/annutacon_studio_{stem}.wav", "duration": round(duration, 2)})

    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"annutacon_studio: {len(entries)} files, {total_hours:.0f}h → {manifest_path}")
    return len(entries)


def import_annutacon_remaining(base_dir: Path) -> int:
    """Import annutacon_remaining (remaining Annutacon conversations).

    Same format as annutacon: data_stereo/{id}.wav + {id}.json
    """
    src_audio = base_dir / "data_stereo"
    if not src_audio.exists():
        print(f"[ERROR] {src_audio} not found")
        return 0

    audio_dir = ensure_dir(base_dir / "audio")
    manifest_path = base_dir / "manifest.jsonl"

    durations = {}
    dataset_jsonl = base_dir / "dataset.jsonl"
    if dataset_jsonl.exists():
        with open(dataset_jsonl) as f:
            for line in f:
                d = json.loads(line)
                stem = Path(d["path"]).stem
                durations[stem] = d["duration"]

    entries = []
    for wav_path in tqdm(sorted(src_audio.glob("*.wav")), desc="annutacon_remaining"):
        stem = wav_path.stem
        json_path = src_audio / f"{stem}.json"

        dst_wav = audio_dir / f"annutacon_remaining_{stem}.wav"
        dst_json = audio_dir / f"annutacon_remaining_{stem}.json"

        _link_or_copy(wav_path, dst_wav)
        if json_path.exists():
            _link_or_copy(json_path, dst_json)

        duration = durations.get(stem) or get_audio_duration(wav_path)
        entries.append({"path": f"audio/annutacon_remaining_{stem}.wav", "duration": round(duration, 2)})

    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"annutacon_remaining: {len(entries)} files, {total_hours:.0f}h → {manifest_path}")
    return len(entries)


IMPORTERS = {
    "annutacon": ("data/external/annutacon", import_annutacon),
    "annutacon_studio": ("data/external/annutacon_studio", import_annutacon_studio),
    "annutacon_remaining": ("data/external/annutacon_remaining", import_annutacon_remaining),
    "fisher": ("data/external/fisher", import_fisher),
}


def main():
    parser = argparse.ArgumentParser(description="Import external datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--list", action="store_true", help="Show dataset status")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking (use when source is on local NVMe)")
    args = parser.parse_args()

    global _USE_COPY
    _USE_COPY = args.copy

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

    if args.dataset == "all":
        # Import all registered datasets that have source data
        for name, (default_dir, importer) in IMPORTERS.items():
            d = Path(default_dir)
            manifest = d / "manifest.jsonl"
            if d.exists() and not manifest.exists():
                print(f"\n=== Importing {name} ===")
                importer(d)
            elif manifest.exists():
                print(f"[SKIP] {name}: manifest already exists")
            else:
                print(f"[SKIP] {name}: dir not found ({d})")
        return

    if args.dataset in IMPORTERS:
        default_dir, importer = IMPORTERS[args.dataset]
        importer(Path(default_dir))
    else:
        # Try generic import for any dataset in data/external/{name}/
        generic_dir = Path(f"data/external/{args.dataset}")
        if generic_dir.exists():
            print(f"Using generic importer for {args.dataset}")
            import_generic(generic_dir)
        else:
            print(f"Unknown dataset: {args.dataset}")
            print(f"  Known: {list(IMPORTERS.keys())}")
            print(f"  Or place data in: data/external/{args.dataset}/raw/*.wav")
            return


if __name__ == "__main__":
    main()
