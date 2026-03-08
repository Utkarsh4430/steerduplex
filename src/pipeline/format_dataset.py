"""Phase 5: Format data for moshi-finetune.

Creates:
1. manifest_train.jsonl + manifest_eval.jsonl (with configurable split)
2. Runs Whisper annotation for word-level alignments
3. Injects <system> tagged text into alignments during prompt region

Usage:
    python -m pipeline.format_dataset --config configs/generation.yaml
    python -m pipeline.format_dataset --config configs/generation.yaml --skip_whisper
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from pipeline.utils import ensure_dir, get_audio_duration, load_json, load_yaml, save_json


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def build_alignments_with_system_prompt(
    metadata: dict,
    whisper_alignments: list | None = None,
) -> list:
    alignments = []
    system_prompt = metadata.get("system_prompt", "")
    if system_prompt:
        prompt_text = wrap_with_system_tags(system_prompt)
        prompt_end = metadata.get("prompt_end_sec", 0.0)
        words = prompt_text.split()
        if words and prompt_end > 0:
            duration_per_word = prompt_end / len(words)
            for i, word in enumerate(words):
                start = round(i * duration_per_word, 3)
                end = round((i + 1) * duration_per_word, 3)
                alignments.append([word, [start, end], "SPEAKER_MAIN"])

    if whisper_alignments:
        alignments.extend(whisper_alignments)

    return alignments


def run_whisper_annotation(manifest_path: Path, lang: str = "en", whisper_model: str = "medium") -> bool:
    annotate_script = Path(__file__).parent.parent / "vendor" / "moshi-finetune" / "annotate.py"
    if not annotate_script.exists():
        print(f"[WARN] annotate.py not found at {annotate_script}. Skipping Whisper.")
        return False

    cmd = [sys.executable, str(annotate_script), str(manifest_path), "--lang", lang, "--whisper_model", whisper_model, "--local"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] annotate.py failed: {result.stderr[:500]}")
        return False
    return True


def format_single(wav_path: Path, meta_path: Path, output_dir: Path) -> dict | None:
    metadata = load_json(meta_path)
    out_wav = output_dir / "audio" / wav_path.name
    ensure_dir(out_wav.parent)
    if not out_wav.exists():
        shutil.copy2(wav_path, out_wav)

    # Check for whisper annotation
    whisper_json = out_wav.with_suffix(".json")
    whisper_alignments = None
    if whisper_json.exists():
        whisper_data = load_json(whisper_json)
        whisper_alignments = whisper_data.get("alignments", [])

    alignments = build_alignments_with_system_prompt(metadata, whisper_alignments)

    transcript_data = {
        "alignments": alignments,
        "text_conditions": None,
        "_metadata": {
            "category": metadata.get("category", ""),
            "data_type": metadata.get("data_type", "standard"),
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
    parser.add_argument("--assembled_dir", type=str, default=None)
    parser.add_argument("--skip_whisper", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    full_cfg = load_yaml(args.config)
    cfg = full_cfg["dataset"]
    quality_cfg = full_cfg["quality"]

    assembled_dir = Path(args.assembled_dir or full_cfg["assembly"]["output_dir"])
    output_dir = ensure_dir(cfg["output_dir"])
    audio_dir = ensure_dir(output_dir / "audio")

    wav_files = sorted(assembled_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    # Step 1: Copy WAVs
    print("=== Step 1: Copying audio ===")
    for wav_path in tqdm(wav_files, desc="Copying"):
        dest = audio_dir / wav_path.name
        if not dest.exists():
            shutil.copy2(wav_path, dest)

    # Step 2: Whisper annotation
    if not args.skip_whisper:
        print("\n=== Step 2: Whisper annotation ===")
        temp_manifest = output_dir / "_temp_manifest.jsonl"
        with open(temp_manifest, "w") as f:
            for wav_path in sorted(audio_dir.glob("*.wav")):
                # Skip if already annotated
                if wav_path.with_suffix(".json").exists():
                    continue
                duration = get_audio_duration(wav_path)
                f.write(json.dumps({"path": str(wav_path), "duration": round(duration, 2)}) + "\n")
        if temp_manifest.stat().st_size > 0:
            run_whisper_annotation(temp_manifest, "en", quality_cfg.get("whisper_model", "medium"))

    # Step 3: Build final dataset with train/eval split
    print("\n=== Step 3: Building dataset ===")
    entries = []
    for wav_path in tqdm(wav_files, desc="Formatting"):
        meta_path = assembled_dir / f"{wav_path.stem}_meta.json"
        if not meta_path.exists():
            continue
        entry = format_single(wav_path, meta_path, output_dir)
        if entry:
            entries.append(entry)

    # Split into train/eval
    eval_ratio = cfg.get("eval_split_ratio", 0.05)
    random.shuffle(entries)
    n_eval = max(1, int(len(entries) * eval_ratio))
    eval_entries = entries[:n_eval]
    train_entries = entries[n_eval:]

    # Write manifests
    train_manifest = output_dir / "manifest_train.jsonl"
    eval_manifest = output_dir / "manifest_eval.jsonl"
    for path, data in [(train_manifest, train_entries), (eval_manifest, eval_entries)]:
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    # Also write combined manifest for compatibility
    combined = output_dir / "manifest.jsonl"
    with open(combined, "w") as f:
        for entry in train_entries + eval_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDataset ready:")
    print(f"  Train: {len(train_entries)} conversations → {train_manifest}")
    print(f"  Eval:  {len(eval_entries)} conversations → {eval_manifest}")
    print(f"  Audio: {audio_dir}")


if __name__ == "__main__":
    main()
