"""Phase 4: Assemble 2-channel (stereo) audio for moshi-finetune.

Combines per-turn audio into stereo WAV:
  - Left channel (ch0): Assistant/Moshi audio
  - Right channel (ch1): User audio

Includes system prompt prepending:
  - Voice prompt (3-10s) prepended to assistant channel
  - 440Hz sine marker (200ms) after voice prompt
  - Silence on user channel during prompt region

Usage:
    python -m pipeline.assemble_channels \
        --config configs/generation.yaml \
        --synth_dir data/tts_audio
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from pipeline.utils import ensure_dir, load_json, load_yaml, save_json, set_seed


def generate_sine(freq_hz: float, duration_sec: float, sample_rate: int) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(duration_sec * sample_rate)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def load_and_resample(audio_path: str, target_sr: int) -> np.ndarray:
    """Load audio and resample to target rate."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel if stereo

    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def build_system_prompt_audio(
    voice_ref_path: str | None,
    voice_prompt_duration: float,
    sine_freq: float,
    sine_duration: float,
    sample_rate: int,
) -> np.ndarray:
    """Build the system prompt audio segment for the assistant channel.

    Structure: [voice_prompt][sine_marker][silence_500ms]

    This is prepended to the assistant channel before the conversation.
    The user channel gets silence for the same duration.
    """
    segments = []

    # Voice prompt (clipped/padded to target duration)
    if voice_ref_path and Path(voice_ref_path).exists():
        ref_audio = load_and_resample(voice_ref_path, sample_rate)
        target_samples = int(voice_prompt_duration * sample_rate)
        if len(ref_audio) > target_samples:
            ref_audio = ref_audio[:target_samples]
        elif len(ref_audio) < target_samples:
            ref_audio = np.pad(ref_audio, (0, target_samples - len(ref_audio)))
        segments.append(ref_audio)
    else:
        # No voice ref: use silence
        segments.append(np.zeros(int(voice_prompt_duration * sample_rate), dtype=np.float32))

    # Sine marker
    segments.append(generate_sine(sine_freq, sine_duration, sample_rate))

    # Brief silence after marker
    segments.append(np.zeros(int(0.5 * sample_rate), dtype=np.float32))

    return np.concatenate(segments)


def assemble_conversation(
    transcript: dict,
    sample_rate: int,
    silence_range_ms: tuple[int, int],
    barge_in_prob: float,
    barge_in_overlap_ms: tuple[int, int],
    max_duration_sec: float,
    voice_prompt_duration: float,
    sine_freq: float,
    sine_duration_ms: float,
) -> tuple[np.ndarray, dict] | None:
    """Assemble a conversation into stereo audio.

    Returns:
        (stereo_audio, metadata) or None on failure
    """
    turns = transcript["turns"]
    assistant_ref = transcript.get("assistant_voice", {}).get("ref_path")

    # Build system prompt audio (assistant channel only)
    prompt_audio = build_system_prompt_audio(
        voice_ref_path=assistant_ref,
        voice_prompt_duration=voice_prompt_duration,
        sine_freq=sine_freq,
        sine_duration=sine_duration_ms / 1000.0,
        sample_rate=sample_rate,
    )
    prompt_len = len(prompt_audio)

    # Estimate total duration to pre-allocate
    max_samples = int(max_duration_sec * sample_rate)

    # Build timeline
    assistant_track = np.zeros(max_samples, dtype=np.float32)
    user_track = np.zeros(max_samples, dtype=np.float32)

    # Place system prompt
    assistant_track[:prompt_len] = prompt_audio

    cursor = prompt_len  # current position in samples
    turn_timestamps = []
    prompt_end_sec = prompt_len / sample_rate

    for i, turn in enumerate(turns):
        audio_path = turn.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            print(f"  [WARN] Missing audio for turn {i}")
            return None

        audio = load_and_resample(audio_path, sample_rate)

        # Add inter-turn gap (or overlap for barge-in)
        if i > 0:
            if random.random() < barge_in_prob:
                # Barge-in: overlap with previous turn
                overlap_ms = random.randint(barge_in_overlap_ms[0], barge_in_overlap_ms[1])
                cursor -= int(overlap_ms / 1000.0 * sample_rate)
                cursor = max(cursor, prompt_len)  # don't overlap into prompt
            else:
                # Normal gap
                gap_ms = random.randint(silence_range_ms[0], silence_range_ms[1])
                cursor += int(gap_ms / 1000.0 * sample_rate)

        # Check if we'd exceed max duration
        if cursor + len(audio) > max_samples:
            # Truncate here
            remaining = max_samples - cursor
            if remaining > sample_rate:  # at least 1 second
                audio = audio[:remaining]
            else:
                break

        # Place audio on correct channel
        end = cursor + len(audio)
        turn_start_sec = cursor / sample_rate
        turn_end_sec = end / sample_rate

        if turn["role"] == "assistant":
            assistant_track[cursor:end] += audio
        else:
            user_track[cursor:end] += audio

        turn_timestamps.append(
            {
                "role": turn["role"],
                "start_sec": round(turn_start_sec, 3),
                "end_sec": round(turn_end_sec, 3),
                "text": turn["text"],
            }
        )

        cursor = end

    # Trim to actual length
    actual_len = min(cursor, max_samples)
    assistant_track = assistant_track[:actual_len]
    user_track = user_track[:actual_len]

    # Normalize channels
    for track in [assistant_track, user_track]:
        peak = np.abs(track).max()
        if peak > 0:
            track *= 0.95 / peak

    # Stereo: left=assistant, right=user
    stereo = np.stack([assistant_track, user_track], axis=-1)

    metadata = {
        "id": transcript["id"],
        "duration_sec": round(actual_len / sample_rate, 3),
        "prompt_end_sec": round(prompt_end_sec, 3),
        "num_turns": len(turn_timestamps),
        "turn_timestamps": turn_timestamps,
        "system_prompt": transcript.get("system_prompt", ""),
        "system_prompt_tts_instruct": transcript.get("system_prompt_tts_instruct", ""),
        "category": transcript.get("category", ""),
        "assistant_voice_id": transcript.get("assistant_voice", {}).get("id", ""),
        "user_voice_id": transcript.get("user_voice", {}).get("id", ""),
    }

    return stereo, metadata


def main():
    parser = argparse.ArgumentParser(description="Assemble 2-channel audio")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--synth_dir", type=str, default="data/tts_audio")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["assembly"]
    set_seed(args.seed)

    output_dir = ensure_dir(cfg["output_dir"])
    synth_dir = Path(args.synth_dir)
    total = 0
    failed = 0

    for cat_dir in sorted(synth_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.category and cat_dir.name != args.category:
            continue

        synth_files = sorted(cat_dir.glob("*_synth.json"))
        print(f"\n=== Assembling {len(synth_files)} conversations for {cat_dir.name} ===")

        for synth_path in tqdm(synth_files, desc=cat_dir.name):
            transcript = load_json(synth_path)

            # Skip if quality check failed
            if not transcript.get("quality_passed", True):
                failed += 1
                continue

            result = assemble_conversation(
                transcript=transcript,
                sample_rate=cfg["sample_rate"],
                silence_range_ms=tuple(cfg["silence_range_ms"]),
                barge_in_prob=cfg["barge_in_prob"],
                barge_in_overlap_ms=tuple(cfg["barge_in_overlap_ms"]),
                max_duration_sec=cfg["max_duration_sec"],
                voice_prompt_duration=cfg["voice_prompt_duration_sec"],
                sine_freq=cfg["sine_marker_freq_hz"],
                sine_duration_ms=cfg["sine_marker_duration_ms"],
            )

            if result is None:
                failed += 1
                continue

            stereo, metadata = result
            conv_id = transcript["id"]

            # Save stereo WAV
            wav_path = output_dir / f"{conv_id}.wav"
            sf.write(str(wav_path), stereo, cfg["sample_rate"])

            # Save metadata
            save_json(metadata, output_dir / f"{conv_id}_meta.json")
            total += 1

    print(f"\nAssembled {total} conversations ({failed} failed/skipped)")


if __name__ == "__main__":
    main()
