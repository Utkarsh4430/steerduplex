"""Phase 2: Assign voice references to conversations.

Assistant voices: Qwen3-TTS CustomVoice presets (speaker name + instruct).
User voices: Qwen3-TTS Base model voice cloning (ref_audio + ref_text).

Resumable: skips already-assigned conversations.

Usage:
    python -m pipeline.assign_voices --config configs/generation.yaml
"""

import argparse
import random
from pathlib import Path

from pipeline.utils import ensure_dir, load_json, load_jsonl, load_yaml, save_json, set_seed


def load_assistant_presets(presets_path: str | Path) -> list[dict]:
    """Load assistant voice presets from YAML."""
    data = load_yaml(presets_path)
    return data.get("voices", [])


def load_user_pool(pool_path: str | Path) -> list[dict]:
    """Load user voice pool from JSONL. Skips template entries."""
    pool = load_jsonl(pool_path)
    return [v for v in pool if v.get("id", "").startswith("_") is False and v.get("ref_path")]


def assign_voices(
    transcript: dict,
    assistant_presets: list[dict],
    user_pool: list[dict],
) -> dict:
    """Assign voice references to a conversation."""
    # Assistant: random preset (CustomVoice model — speaker + instruct)
    if assistant_presets:
        voice = random.choice(assistant_presets)
        transcript["assistant_voice"] = {
            "id": voice["id"],
            "speaker": voice["speaker"],
            "gender": voice.get("gender", "unknown"),
            "model": "CustomVoice",
        }
    else:
        transcript["assistant_voice"] = {
            "id": "placeholder",
            "speaker": "Ryan",
            "gender": "male",
            "model": "CustomVoice",
        }

    # User: random from pool (Base model — ref_audio cloning)
    if user_pool:
        voice = random.choice(user_pool)
        transcript["user_voice"] = {
            "id": voice["id"],
            "ref_path": voice["ref_path"],
            "ref_text": voice.get("ref_text", ""),
            "gender": voice.get("gender", "unknown"),
            "model": "Base",
        }
    else:
        # Fallback: use a CustomVoice preset for user too
        transcript["user_voice"] = {
            "id": "preset_user",
            "speaker": "Eric",
            "gender": "male",
            "model": "CustomVoice",
        }

    return transcript


def main():
    parser = argparse.ArgumentParser(description="Assign voices to transcripts")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--transcripts_dir", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["voices"]
    set_seed(args.seed)

    assistant_presets = load_assistant_presets(cfg["assistant_presets"])
    user_pool = load_user_pool(cfg["user_pool"])
    print(f"Assistant presets: {len(assistant_presets)} | User pool: {len(user_pool)}")

    if not user_pool:
        print("[WARN] No user voices found. Will use CustomVoice preset fallback.")

    transcripts_dir = Path(args.transcripts_dir or load_yaml(args.config)["transcript"]["output_dir"])
    output_dir = ensure_dir(cfg["output_dir"])
    total = 0

    for cat_dir in sorted(transcripts_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.category and cat_dir.name != args.category:
            continue

        cat_output = ensure_dir(output_dir / cat_dir.name)

        for transcript_path in sorted(cat_dir.glob("*.json")):
            out_path = cat_output / transcript_path.name
            if out_path.exists():
                continue  # resume

            transcript = load_json(transcript_path)
            transcript = assign_voices(transcript, assistant_presets, user_pool)
            save_json(transcript, out_path)
            total += 1

    print(f"Assigned voices to {total} new conversations")


if __name__ == "__main__":
    main()
