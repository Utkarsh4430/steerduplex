"""Phase 2: Assign voice references to conversations.

Maps each conversation to an assistant voice (from curated pool)
and a user voice (from large random pool).

Usage:
    python -m pipeline.assign_voices \
        --config configs/generation.yaml \
        --transcripts_dir data/transcripts
"""

import argparse
import random
from pathlib import Path

from pipeline.utils import ensure_dir, load_json, load_yaml, save_json, set_seed


def discover_voice_pool(voices_dir: str | Path) -> list[dict]:
    """Discover available voice references in a directory.

    Expects structure:
        voices_dir/
            speaker_001/
                ref.wav        # 3-10s reference clip
                metadata.json  # optional: {gender, accent, age_range, energy}
            speaker_002/
                ref.wav
                ...
    """
    voices = []
    voices_dir = Path(voices_dir)
    if not voices_dir.exists():
        return voices

    for speaker_dir in sorted(voices_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        ref_wav = speaker_dir / "ref.wav"
        if not ref_wav.exists():
            # Try other extensions
            for ext in [".flac", ".mp3", ".ogg"]:
                ref_wav = speaker_dir / f"ref{ext}"
                if ref_wav.exists():
                    break
            else:
                continue

        metadata = {}
        meta_path = speaker_dir / "metadata.json"
        if meta_path.exists():
            metadata = load_json(meta_path)

        voices.append(
            {
                "id": speaker_dir.name,
                "ref_path": str(ref_wav),
                **metadata,
            }
        )

    return voices


def assign_voices_to_conversation(
    transcript: dict,
    assistant_voices: list[dict],
    user_voices: list[dict],
    category_id: str,
) -> dict:
    """Assign voice references to a conversation transcript."""
    # Assistant: round-robin or random from curated pool
    assistant_voice = random.choice(assistant_voices) if assistant_voices else {"id": "default", "ref_path": None}

    # User: random from large pool
    user_voice = random.choice(user_voices) if user_voices else {"id": "default", "ref_path": None}

    # For A5 (accent category): try to match user voice accent if tagged
    if category_id.startswith("A5") and "accent" in transcript.get("sampled_traits", {}):
        target_accent = transcript["sampled_traits"]["accent"]
        matching = [v for v in user_voices if v.get("accent", "").lower() == target_accent.lower()]
        if matching:
            user_voice = random.choice(matching)

    transcript["assistant_voice"] = assistant_voice
    transcript["user_voice"] = user_voice
    return transcript


def main():
    parser = argparse.ArgumentParser(description="Assign voices to transcripts")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--transcripts_dir", type=str, default="data/transcripts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["voices"]
    set_seed(args.seed)

    # Discover voice pools
    assistant_voices = discover_voice_pool(cfg["assistant_voices_dir"])
    user_voices = discover_voice_pool(cfg["user_voices_dir"])
    print(f"Found {len(assistant_voices)} assistant voices, {len(user_voices)} user voices")

    if not assistant_voices:
        print("[WARN] No assistant voices found. Will use placeholder.")
        assistant_voices = [{"id": "placeholder_assistant", "ref_path": None}]
    if not user_voices:
        print("[WARN] No user voices found. Will use placeholder.")
        user_voices = [{"id": "placeholder_user", "ref_path": None}]

    output_dir = ensure_dir(cfg["output_dir"])
    transcripts_dir = Path(args.transcripts_dir)

    total = 0
    for cat_dir in sorted(transcripts_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_id = cat_dir.name
        cat_output = ensure_dir(output_dir / cat_id)

        for transcript_path in sorted(cat_dir.glob("*.json")):
            transcript = load_json(transcript_path)
            transcript = assign_voices_to_conversation(
                transcript, assistant_voices, user_voices, cat_id
            )
            save_json(transcript, cat_output / transcript_path.name)
            total += 1

    print(f"Assigned voices to {total} conversations")


if __name__ == "__main__":
    main()
