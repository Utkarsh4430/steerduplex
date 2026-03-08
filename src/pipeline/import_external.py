"""Import external datasets (Fisher, VoxCeleb, custom) into SteerDuplex format.

Converts pre-existing stereo or mono audio datasets into the moshi-finetune
training format. Handles the full pipeline: audio preparation → Whisper
annotation → manifest generation.

Supported input formats:
  1. Stereo WAV (left=assistant, right=user) — e.g., Fisher
  2. Mono WAV pairs (separate assistant + user files) — e.g., recorded conversations
  3. Single-speaker WAV + transcript — e.g., VoxCeleb, LibriSpeech (user voices only)

Output format (moshi-finetune compatible):
  data/external/<dataset_name>/
  ├── audio/
  │   ├── 00001.wav          # Stereo WAV: left=assistant, right=user
  │   ├── 00001.json         # Whisper word-level alignments
  │   ├── 00002.wav
  │   └── 00002.json
  ├── manifest_train.jsonl   # {"path": "audio/00001.wav", "duration": 45.2}
  └── manifest_eval.jsonl

Usage:
    # Import stereo Fisher conversations
    python -m pipeline.import_external \\
        --input_dir /data/fisher/audio \\
        --dataset_name fisher \\
        --format stereo \\
        --system_prompt "You are a helpful voice assistant."

    # Import mono conversation pairs
    python -m pipeline.import_external \\
        --input_dir /data/conversations \\
        --dataset_name custom_convos \\
        --format mono_pairs \\
        --assistant_suffix "_agent.wav" \\
        --user_suffix "_caller.wav"

    # Import single-speaker clips as user voice pool
    python -m pipeline.import_external \\
        --input_dir /data/voxceleb1/wav \\
        --dataset_name voxceleb \\
        --format voice_pool \\
        --max_clips 1000

    # Skip Whisper annotation (if you'll do it separately)
    python -m pipeline.import_external \\
        --input_dir /data/fisher/audio \\
        --dataset_name fisher \\
        --format stereo \\
        --skip_whisper
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from pipeline.utils import ensure_dir, get_audio_duration


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def import_stereo(
    input_dir: Path,
    output_dir: Path,
    system_prompt: str,
    max_duration_sec: float,
    target_sr: int,
) -> list[dict]:
    """Import stereo WAV files (left=assistant, right=user).

    Expected for: Fisher, pre-assembled conversations, etc.
    """
    audio_dir = ensure_dir(output_dir / "audio")
    entries = []
    wav_files = sorted(input_dir.rglob("*.wav"))

    print(f"Found {len(wav_files)} WAV files in {input_dir}")

    for i, wav_path in enumerate(tqdm(wav_files, desc="Importing stereo")):
        out_name = f"{i:06d}.wav"
        out_path = audio_dir / out_name

        if out_path.exists():
            duration = get_audio_duration(out_path)
            entries.append({"path": f"audio/{out_name}", "duration": round(duration, 2)})
            continue

        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            print(f"  [SKIP] {wav_path}: {e}")
            continue

        # Ensure stereo
        if audio.ndim == 1:
            # Mono → duplicate to both channels
            audio = np.stack([audio, audio], axis=-1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        # Resample if needed
        if sr != target_sr:
            import librosa
            left = librosa.resample(audio[:, 0], orig_sr=sr, target_sr=target_sr)
            right = librosa.resample(audio[:, 1], orig_sr=sr, target_sr=target_sr)
            audio = np.stack([left, right], axis=-1)
            sr = target_sr

        # Truncate
        max_samples = int(max_duration_sec * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        duration = len(audio) / sr
        if duration < 1.0:
            continue

        # Normalize per channel
        for ch in range(2):
            peak = np.abs(audio[:, ch]).max()
            if peak > 0:
                audio[:, ch] *= 0.95 / peak

        sf.write(str(out_path), audio, sr)

        # Write system prompt annotation
        if system_prompt:
            tagged = wrap_with_system_tags(system_prompt)
            annotation = {
                "alignments": [],
                "text_conditions": {
                    "prompt_end_sec": "0.0",
                    "system_prompt": tagged,
                },
                "_metadata": {
                    "source": str(wav_path),
                },
            }
            json_path = out_path.with_suffix(".json")
            if not json_path.exists():
                with open(json_path, "w") as f:
                    json.dump(annotation, f, ensure_ascii=False)

        entries.append({"path": f"audio/{out_name}", "duration": round(duration, 2)})

    return entries


def import_mono_pairs(
    input_dir: Path,
    output_dir: Path,
    assistant_suffix: str,
    user_suffix: str,
    system_prompt: str,
    max_duration_sec: float,
    target_sr: int,
) -> list[dict]:
    """Import paired mono WAV files → assemble into stereo.

    Expects files like: conv_001_agent.wav + conv_001_caller.wav
    """
    audio_dir = ensure_dir(output_dir / "audio")

    # Find pairs
    assistant_files = sorted(input_dir.rglob(f"*{assistant_suffix}"))
    pairs = []
    for af in assistant_files:
        base = str(af).replace(assistant_suffix, "")
        uf = Path(base + user_suffix)
        if uf.exists():
            pairs.append((af, uf))

    print(f"Found {len(pairs)} conversation pairs")
    entries = []

    for i, (af, uf) in enumerate(tqdm(pairs, desc="Importing pairs")):
        out_name = f"{i:06d}.wav"
        out_path = audio_dir / out_name

        if out_path.exists():
            duration = get_audio_duration(out_path)
            entries.append({"path": f"audio/{out_name}", "duration": round(duration, 2)})
            continue

        try:
            a_audio, a_sr = sf.read(str(af), dtype="float32")
            u_audio, u_sr = sf.read(str(uf), dtype="float32")
        except Exception as e:
            print(f"  [SKIP] {af}: {e}")
            continue

        # Make mono
        if a_audio.ndim > 1:
            a_audio = a_audio[:, 0]
        if u_audio.ndim > 1:
            u_audio = u_audio[:, 0]

        # Resample
        if a_sr != target_sr:
            import librosa
            a_audio = librosa.resample(a_audio, orig_sr=a_sr, target_sr=target_sr)
        if u_sr != target_sr:
            import librosa
            u_audio = librosa.resample(u_audio, orig_sr=u_sr, target_sr=target_sr)

        # Match lengths
        max_len = min(max(len(a_audio), len(u_audio)), int(max_duration_sec * target_sr))
        a_audio = np.pad(a_audio, (0, max(0, max_len - len(a_audio))))[:max_len]
        u_audio = np.pad(u_audio, (0, max(0, max_len - len(u_audio))))[:max_len]

        duration = max_len / target_sr
        if duration < 1.0:
            continue

        # Normalize
        for arr in [a_audio, u_audio]:
            peak = np.abs(arr).max()
            if peak > 0:
                arr *= 0.95 / peak

        stereo = np.stack([a_audio, u_audio], axis=-1)
        sf.write(str(out_path), stereo, target_sr)

        entries.append({"path": f"audio/{out_name}", "duration": round(duration, 2)})

    return entries


def import_voice_pool(
    input_dir: Path,
    output_dir: Path,
    max_clips: int,
) -> int:
    """Import single-speaker clips into user voice pool (not training data).

    Scans for WAV files, picks one per speaker directory, writes to pool.jsonl.
    """
    pool_path = output_dir / "pool_import.jsonl"
    voice_audio_dir = ensure_dir(output_dir / "audio")

    # Find speaker directories or individual files
    wav_files = sorted(input_dir.rglob("*.wav"))
    if not wav_files:
        print("No WAV files found")
        return 0

    # Group by parent directory (assumed = speaker)
    speakers: dict[str, list[Path]] = {}
    for wf in wav_files:
        spk = wf.parent.name
        speakers.setdefault(spk, []).append(wf)

    count = 0
    with open(pool_path, "w") as f:
        for spk_id, files in tqdm(sorted(speakers.items())[:max_clips], desc="Voice pool"):
            # Pick the best reference clip (3-10s, prefer ~5s)
            best = None
            best_diff = float("inf")
            for wf in files:
                try:
                    dur = get_audio_duration(wf)
                    if 2.0 <= dur <= 15.0:
                        diff = abs(dur - 5.0)
                        if diff < best_diff:
                            best = wf
                            best_diff = diff
                except Exception:
                    continue

            if best is None:
                best = files[0]

            # Copy to voice pool
            spk_dir = ensure_dir(voice_audio_dir / spk_id)
            dest = spk_dir / "ref.wav"
            if not dest.exists():
                shutil.copy2(best, dest)

            entry = {
                "id": spk_id,
                "ref_path": str(dest),
                "ref_text": "",
                "gender": "unknown",
                "accent": "unknown",
                "age_range": "unknown",
                "energy": "medium",
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(f"\nWrote {count} voice entries to {pool_path}")
    print(f"To use: append to data/voices/user/pool.jsonl")
    return count


def write_manifests(
    entries: list[dict],
    output_dir: Path,
    eval_ratio: float = 0.05,
    seed: int = 42,
):
    """Write train/eval manifest JSONL files."""
    random.seed(seed)
    random.shuffle(entries)

    n_eval = max(1, int(len(entries) * eval_ratio))
    eval_entries = entries[:n_eval]
    train_entries = entries[n_eval:]

    for name, data in [("manifest_train.jsonl", train_entries), ("manifest_eval.jsonl", eval_entries)]:
        path = output_dir / name
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    print(f"Train: {len(train_entries)} | Eval: {len(eval_entries)}")
    print(f"Manifests: {output_dir}/manifest_{{train,eval}}.jsonl")
    return train_entries, eval_entries


def run_whisper(output_dir: Path, whisper_model: str = "medium"):
    """Run Whisper annotation on all un-annotated audio files."""
    audio_dir = output_dir / "audio"
    unannotated = []
    for wav in sorted(audio_dir.glob("*.wav")):
        if not wav.with_suffix(".json").exists():
            unannotated.append(wav)

    if not unannotated:
        print("All files already annotated")
        return

    print(f"Running Whisper on {len(unannotated)} files...")
    temp_manifest = output_dir / "_whisper_manifest.jsonl"
    with open(temp_manifest, "w") as f:
        for wav in unannotated:
            dur = get_audio_duration(wav)
            f.write(json.dumps({"path": str(wav), "duration": round(dur, 2)}) + "\n")

    cmd = [
        sys.executable, "-m", "training.annotate",
        str(temp_manifest), "--lang", "en",
        "--whisper_model", whisper_model, "--local",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
    temp_manifest.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Import external datasets into SteerDuplex format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data format reference:
  moshi-finetune expects:
  ├── audio/
  │   ├── 00001.wav    # Stereo: left=assistant (ch0), right=user (ch1)
  │   └── 00001.json   # Word-level alignments from Whisper
  └── manifest.jsonl   # {"path": "audio/00001.wav", "duration": 45.2}

  The .json alignment file format:
  {
    "alignments": [
      ["Hello", [0.5, 0.8], "SPEAKER_MAIN"],
      ["how", [0.8, 1.0], "SPEAKER_MAIN"],
      ...
    ],
    "text_conditions": null,
    "_metadata": {
      "system_prompt": "<system> You are a helpful assistant. <system>",
      "prompt_end_sec": 0.0
    }
  }
        """,
    )

    parser.add_argument("--input_dir", type=str, required=True, help="Input audio directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name for this dataset (used as output subdir)")
    parser.add_argument("--format", type=str, required=True, choices=["stereo", "mono_pairs", "voice_pool"],
                        help="Input format type")
    parser.add_argument("--output_base", type=str, default="data/external", help="Base output directory")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful voice assistant.",
                        help="System prompt to inject (for stereo/mono_pairs)")
    parser.add_argument("--max_duration", type=float, default=100.0, help="Max audio duration in seconds")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate")
    parser.add_argument("--eval_ratio", type=float, default=0.05, help="Eval split ratio")
    parser.add_argument("--skip_whisper", action="store_true", help="Skip Whisper annotation")
    parser.add_argument("--whisper_model", type=str, default="medium", help="Whisper model size")

    # mono_pairs options
    parser.add_argument("--assistant_suffix", type=str, default="_agent.wav")
    parser.add_argument("--user_suffix", type=str, default="_caller.wav")

    # voice_pool options
    parser.add_argument("--max_clips", type=int, default=1000, help="Max voice clips to import")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = ensure_dir(Path(args.output_base) / args.dataset_name)
    print(f"Importing {args.format} data from {input_dir} → {output_dir}")

    if args.format == "voice_pool":
        import_voice_pool(input_dir, output_dir, max_clips=args.max_clips)
        return

    if args.format == "stereo":
        entries = import_stereo(
            input_dir, output_dir,
            system_prompt=args.system_prompt,
            max_duration_sec=args.max_duration,
            target_sr=args.sample_rate,
        )
    elif args.format == "mono_pairs":
        entries = import_mono_pairs(
            input_dir, output_dir,
            assistant_suffix=args.assistant_suffix,
            user_suffix=args.user_suffix,
            system_prompt=args.system_prompt,
            max_duration_sec=args.max_duration,
            target_sr=args.sample_rate,
        )

    if not entries:
        print("No entries imported")
        return

    # Run Whisper annotation
    if not args.skip_whisper:
        run_whisper(output_dir, args.whisper_model)

    # Write manifests
    write_manifests(entries, output_dir, eval_ratio=args.eval_ratio, seed=args.seed)

    print(f"\nDone! To include in training, add to your training config:")
    print(f"  train_data: \"{output_dir}/manifest_train.jsonl\"")
    print(f"  eval_data: \"{output_dir}/manifest_eval.jsonl\"")
    print(f"\nOr merge with existing data:")
    print(f"  train_data: \"data/formatted/manifest_train.jsonl:1.0,{output_dir}/manifest_train.jsonl:0.5\"")


if __name__ == "__main__":
    main()
