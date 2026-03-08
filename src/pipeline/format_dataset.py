"""Phase 5: Format data for moshi-finetune.

Takes assembled stereo WAVs and produces:
1. A manifest.jsonl with {path, duration} entries
2. Runs Whisper annotation to generate per-file .json transcripts
3. Injects system prompt text into the transcript alignments

The key insight from PersonaPlex: system prompts are baked into the
text token stream using <system> tags. During training, moshi-finetune
will learn to condition on these tokens.

For the pilot, we prepend the system prompt as text in the alignments
at the beginning (during the voice prompt region). The model sees:
  - Audio: voice_prompt + sine + silence + conversation
  - Text: <system> prompt text <system> + conversation transcript

Usage:
    python -m pipeline.format_dataset \
        --config configs/generation.yaml \
        --assembled_dir data/assembled
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from pipeline.utils import ensure_dir, get_audio_duration, load_json, load_yaml, save_json


def wrap_with_system_tags(text: str) -> str:
    """Wrap text with system tags (PersonaPlex convention).

    The Moshi/Helium backbone learns to condition on <system> delimiters.
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def build_alignments_with_system_prompt(
    metadata: dict,
    whisper_alignments: list | None = None,
) -> list:
    """Build alignment list with system prompt injected at the start.

    The system prompt text is placed in the alignment timeline
    during the voice prompt region (before conversation starts).
    """
    alignments = []

    # Inject system prompt as aligned text during prompt region
    system_prompt = metadata.get("system_prompt", "")
    if system_prompt:
        prompt_text = wrap_with_system_tags(system_prompt)
        prompt_end = metadata.get("prompt_end_sec", 0.0)

        # Place system prompt words across the prompt region
        words = prompt_text.split()
        if words and prompt_end > 0:
            duration_per_word = prompt_end / len(words)
            for i, word in enumerate(words):
                start = round(i * duration_per_word, 3)
                end = round((i + 1) * duration_per_word, 3)
                alignments.append([word, [start, end], "SPEAKER_MAIN"])

    # Append whisper-generated alignments for the actual conversation
    if whisper_alignments:
        alignments.extend(whisper_alignments)

    return alignments


def create_manifest(assembled_dir: Path, output_dir: Path) -> Path:
    """Create manifest.jsonl from assembled WAV files."""
    manifest_path = output_dir / "manifest.jsonl"

    with open(manifest_path, "w") as f:
        for wav_path in sorted(assembled_dir.glob("*.wav")):
            duration = get_audio_duration(wav_path)
            # Use relative path from output_dir
            rel_path = str(wav_path.relative_to(output_dir)) if wav_path.is_relative_to(output_dir) else str(wav_path)
            entry = {"path": rel_path, "duration": round(duration, 2)}
            f.write(json.dumps(entry) + "\n")

    return manifest_path


def run_whisper_annotation(manifest_path: Path, lang: str = "en", whisper_model: str = "medium"):
    """Run moshi-finetune's annotate.py to generate .json transcripts."""
    annotate_script = Path(__file__).parent.parent / "vendor" / "moshi-finetune" / "annotate.py"

    if not annotate_script.exists():
        print(f"[WARN] annotate.py not found at {annotate_script}")
        print("  Skipping Whisper annotation. Will use metadata-based alignments only.")
        return False

    cmd = [
        sys.executable,
        str(annotate_script),
        str(manifest_path),
        "--lang", lang,
        "--whisper_model", whisper_model,
        "--local",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[WARN] annotate.py failed: {result.stderr}")
        return False

    return True


def format_single(
    wav_path: Path,
    meta_path: Path,
    output_dir: Path,
    use_whisper: bool = True,
) -> dict | None:
    """Format a single conversation for moshi-finetune."""
    metadata = load_json(meta_path)

    # Copy WAV to output dir (or symlink)
    out_wav = output_dir / "audio" / wav_path.name
    ensure_dir(out_wav.parent)
    if not out_wav.exists():
        import shutil
        shutil.copy2(wav_path, out_wav)

    # Check if whisper annotation exists
    whisper_json = out_wav.with_suffix(".json")
    whisper_alignments = None
    if whisper_json.exists():
        whisper_data = load_json(whisper_json)
        whisper_alignments = whisper_data.get("alignments", [])

    # Build final alignments with system prompt
    alignments = build_alignments_with_system_prompt(
        metadata=metadata,
        whisper_alignments=whisper_alignments,
    )

    # Write final .json transcript
    transcript_data = {
        "alignments": alignments,
        # Store system prompt info for potential future use with text_conditions
        "text_conditions": None,
        # Metadata for tracking
        "_metadata": {
            "category": metadata.get("category", ""),
            "system_prompt": metadata.get("system_prompt", ""),
            "prompt_end_sec": metadata.get("prompt_end_sec", 0.0),
            "assistant_voice_id": metadata.get("assistant_voice_id", ""),
        },
    }
    save_json(transcript_data, out_wav.with_suffix(".json"))

    return {
        "path": f"audio/{wav_path.name}",
        "duration": metadata["duration_sec"],
    }


def main():
    parser = argparse.ArgumentParser(description="Format dataset for moshi-finetune")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--assembled_dir", type=str, default="data/assembled")
    parser.add_argument("--skip_whisper", action="store_true", help="Skip Whisper annotation")
    args = parser.parse_args()

    cfg = load_yaml(args.config)["dataset"]
    quality_cfg = load_yaml(args.config)["quality"]
    output_dir = ensure_dir(cfg["output_dir"])
    assembled_dir = Path(args.assembled_dir)

    # Step 1: Copy WAVs and create initial manifest for Whisper
    print("=== Step 1: Preparing audio files ===")
    audio_dir = ensure_dir(output_dir / "audio")
    wav_files = sorted(assembled_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    import shutil
    for wav_path in tqdm(wav_files, desc="Copying"):
        dest = audio_dir / wav_path.name
        if not dest.exists():
            shutil.copy2(wav_path, dest)

    # Step 2: Create temporary manifest for Whisper annotation
    if not args.skip_whisper:
        print("\n=== Step 2: Running Whisper annotation ===")
        # Create manifest pointing to audio/ subdirectory
        temp_manifest = output_dir / "_temp_manifest.jsonl"
        with open(temp_manifest, "w") as f:
            for wav_path in sorted(audio_dir.glob("*.wav")):
                duration = get_audio_duration(wav_path)
                entry = {"path": str(wav_path), "duration": round(duration, 2)}
                f.write(json.dumps(entry) + "\n")

        run_whisper_annotation(
            temp_manifest,
            lang="en",
            whisper_model=quality_cfg["whisper_model"],
        )

    # Step 3: Build final transcripts with system prompts and manifest
    print("\n=== Step 3: Building final dataset ===")
    manifest_entries = []

    for wav_path in tqdm(wav_files, desc="Formatting"):
        meta_path = assembled_dir / f"{wav_path.stem}_meta.json"
        if not meta_path.exists():
            print(f"  [WARN] Missing metadata for {wav_path.name}")
            continue

        entry = format_single(wav_path, meta_path, output_dir, use_whisper=not args.skip_whisper)
        if entry:
            manifest_entries.append(entry)

    # Write final manifest
    manifest_path = Path(cfg["manifest_path"])
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDataset ready: {len(manifest_entries)} conversations")
    print(f"Manifest: {manifest_path}")
    print(f"Audio dir: {audio_dir}")


if __name__ == "__main__":
    main()
