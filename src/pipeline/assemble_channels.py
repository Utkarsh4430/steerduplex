"""Phase 4: Assemble 2-channel (stereo) audio for moshi-finetune.

Left channel (ch0): Assistant/Moshi audio
Right channel (ch1): User audio

Includes system prompt prepending on assistant channel.
Resumable: skips already-assembled conversations.

Usage:
    python -m pipeline.assemble_channels --config configs/generation.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from pipeline.utils import ensure_dir, load_json, load_yaml, save_json, set_seed


def generate_sine(freq_hz: float, duration_sec: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(duration_sec * sample_rate)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def load_and_resample(audio_path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
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
    """Build [voice_ref][sine_marker][silence] for assistant channel."""
    segments = []
    target_samples = int(voice_prompt_duration * sample_rate)

    if voice_ref_path and Path(voice_ref_path).exists():
        ref_audio = load_and_resample(voice_ref_path, sample_rate)
        if len(ref_audio) > target_samples:
            ref_audio = ref_audio[:target_samples]
        elif len(ref_audio) < target_samples:
            ref_audio = np.pad(ref_audio, (0, target_samples - len(ref_audio)))
        segments.append(ref_audio)
    else:
        segments.append(np.zeros(target_samples, dtype=np.float32))

    segments.append(generate_sine(sine_freq, sine_duration, sample_rate))
    segments.append(np.zeros(int(0.5 * sample_rate), dtype=np.float32))
    return np.concatenate(segments)


def _apply_fade_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply a linear fade-out to the last `fade_samples` of audio."""
    if fade_samples <= 0 or len(audio) == 0:
        return audio
    fade_samples = min(fade_samples, len(audio))
    fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    audio = audio.copy()
    audio[-fade_samples:] *= fade
    return audio


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
    # Role-aware gap overrides (ms). If set, these take priority over silence_range_ms.
    user_to_assistant_gap_ms: tuple[int, int] | None = None,
    assistant_to_user_gap_ms: tuple[int, int] | None = None,
    # Barge-in truncation: fade out assistant audio when user interrupts
    barge_in_fade_ms: float = 150.0,
    # User silence: probability and duration range for inserting extended silence
    # before a user turn (trains model to handle pauses / "are you still there?")
    user_silence_prob: float = 0.0,
    user_silence_range_ms: tuple[int, int] = (2000, 6000),
) -> tuple[np.ndarray, dict] | None:
    """Assemble stereo WAV from per-turn audio files."""
    turns = transcript["turns"]

    # For CustomVoice assistant, we don't have a ref_path file.
    # Generate a voice prompt segment from the first assistant turn if available.
    assistant_voice = transcript.get("assistant_voice", {})
    voice_ref = None
    if assistant_voice.get("model") == "Base":
        voice_ref = assistant_voice.get("ref_path")
    # For CustomVoice: no ref file, use silence for voice prompt region
    # The system prompt text still gets injected in format_dataset

    prompt_audio = build_system_prompt_audio(
        voice_ref_path=voice_ref,
        voice_prompt_duration=voice_prompt_duration,
        sine_freq=sine_freq,
        sine_duration=sine_duration_ms / 1000.0,
        sample_rate=sample_rate,
    )
    prompt_len = len(prompt_audio)
    max_samples = int(max_duration_sec * sample_rate)

    assistant_track = np.zeros(max_samples, dtype=np.float32)
    user_track = np.zeros(max_samples, dtype=np.float32)
    assistant_track[:prompt_len] = prompt_audio

    # Fill user channel with 440 Hz sine during system prompt region
    # (PersonaPlex pattern — stable conditioning signal that tells the model
    # "we're in prompt mode, don't start responding yet")
    user_sine = generate_sine(440.0, prompt_len / sample_rate, sample_rate)
    user_track[:prompt_len] = user_sine[:prompt_len]

    cursor = prompt_len
    turn_timestamps = []

    for i, turn in enumerate(turns):
        audio_path = turn.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            print(f"  [WARN] Missing audio for turn {i}")
            return None

        audio = load_and_resample(audio_path, sample_rate)

        if i > 0:
            prev_role = turns[i - 1]["role"]
            curr_role = turn["role"]

            # Barge-in: user interrupts assistant — overlap + truncate assistant
            if prev_role == "assistant" and curr_role == "user" and random.random() < barge_in_prob:
                overlap_ms = random.randint(barge_in_overlap_ms[0], barge_in_overlap_ms[1])
                overlap_samples = int(overlap_ms / 1000.0 * sample_rate)
                cursor -= overlap_samples
                cursor = max(cursor, prompt_len)

                # Truncate + fade out the assistant audio at the barge-in point
                # so the model learns: when user starts speaking, stop talking
                fade_samples = int(barge_in_fade_ms / 1000.0 * sample_rate)
                trunc_point = cursor + overlap_samples  # where user voice starts
                fade_end = min(trunc_point + fade_samples, max_samples)
                if trunc_point < max_samples:
                    # Apply fade from barge-in point, then silence the rest
                    fade_len = fade_end - trunc_point
                    if fade_len > 0:
                        fade_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                        assistant_track[trunc_point:fade_end] *= fade_curve
                    # Zero out everything after the fade
                    if fade_end < max_samples:
                        prev_end = turn_timestamps[-1]["end_sec"] * sample_rate if turn_timestamps else max_samples
                        zero_end = min(int(prev_end), max_samples)
                        assistant_track[fade_end:zero_end] = 0.0

            # Extended user silence: user pauses for several seconds before speaking
            # Trains the model to handle "dead air" without hallucinating
            elif curr_role == "user" and prev_role == "assistant" and random.random() < user_silence_prob:
                silence_ms = random.randint(user_silence_range_ms[0], user_silence_range_ms[1])
                cursor += int(silence_ms / 1000.0 * sample_rate)

            else:
                # Role-aware gaps: assistant should respond quickly,
                # user gets natural think time
                if prev_role == "user" and curr_role == "assistant" and user_to_assistant_gap_ms:
                    gap_range = user_to_assistant_gap_ms
                elif prev_role == "assistant" and curr_role == "user" and assistant_to_user_gap_ms:
                    gap_range = assistant_to_user_gap_ms
                else:
                    gap_range = silence_range_ms
                gap_ms = random.randint(gap_range[0], gap_range[1])
                cursor += int(gap_ms / 1000.0 * sample_rate)

        if cursor + len(audio) > max_samples:
            remaining = max_samples - cursor
            if remaining > sample_rate:
                audio = audio[:remaining]
            else:
                break

        end = cursor + len(audio)
        if turn["role"] == "assistant":
            assistant_track[cursor:end] += audio
        else:
            user_track[cursor:end] += audio

        turn_timestamps.append({
            "role": turn["role"],
            "start_sec": round(cursor / sample_rate, 3),
            "end_sec": round(end / sample_rate, 3),
            "text": turn["text"],
        })
        cursor = end

    actual_len = min(cursor, max_samples)
    assistant_track = assistant_track[:actual_len]
    user_track = user_track[:actual_len]

    for track in [assistant_track, user_track]:
        peak = np.abs(track).max()
        if peak > 0:
            track *= 0.95 / peak

    stereo = np.stack([assistant_track, user_track], axis=-1)

    metadata = {
        "id": transcript["id"],
        "category": transcript.get("category", ""),
        "data_type": transcript.get("data_type", "standard"),
        "duration_sec": round(actual_len / sample_rate, 3),
        "prompt_end_sec": round(prompt_len / sample_rate, 3),
        "num_turns": len(turn_timestamps),
        "turn_timestamps": turn_timestamps,
        "system_prompt": transcript.get("system_prompt", ""),
        "system_prompt_rephrased": transcript.get("system_prompt_rephrased", ""),
        "assistant_voice_id": transcript.get("assistant_voice", {}).get("id", ""),
        "user_voice_id": transcript.get("user_voice", {}).get("id", ""),
    }
    return stereo, metadata


def main():
    parser = argparse.ArgumentParser(description="Assemble 2-channel audio")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--synth_dir", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["assembly"]
    set_seed(args.seed)

    from pipeline.distributed import is_done, release_claim, try_claim

    synth_dir = Path(args.synth_dir or load_yaml(args.config)["tts"]["output_dir"])
    output_dir = ensure_dir(cfg["output_dir"])
    claims_dir = ensure_dir(output_dir / ".claims")
    total = 0
    failed = 0

    for cat_dir in sorted(synth_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.category and cat_dir.name != args.category:
            continue

        cat_claims = ensure_dir(claims_dir / cat_dir.name)
        synth_files = sorted(cat_dir.glob("*_synth.json"))

        remaining = []
        for f in synth_files:
            transcript = load_json(f)
            conv_id = transcript.get("id", "")
            meta_path = output_dir / f"{conv_id}_meta.json"
            if not is_done(meta_path):
                remaining.append(f)

        if not remaining:
            print(f"[SKIP] {cat_dir.name}: all done")
            continue

        print(f"\n=== {cat_dir.name}: {len(remaining)} remaining ===")

        for synth_path in tqdm(remaining, desc=cat_dir.name):
            transcript = load_json(synth_path)
            conv_id = transcript.get("id", synth_path.stem)

            meta_path = output_dir / f"{conv_id}_meta.json"
            if is_done(meta_path):
                continue

            claim_path = cat_claims / f"{conv_id}.claim"
            if not try_claim(claim_path):
                continue  # another node claimed it

            u2a_gap = tuple(cfg["user_to_assistant_gap_ms"]) if "user_to_assistant_gap_ms" in cfg else None
            a2u_gap = tuple(cfg["assistant_to_user_gap_ms"]) if "assistant_to_user_gap_ms" in cfg else None

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
                user_to_assistant_gap_ms=u2a_gap,
                assistant_to_user_gap_ms=a2u_gap,
                barge_in_fade_ms=cfg.get("barge_in_fade_ms", 150.0),
                user_silence_prob=cfg.get("user_silence_prob", 0.0),
                user_silence_range_ms=tuple(cfg.get("user_silence_range_ms", [2000, 6000])),
            )

            if result is None:
                release_claim(claim_path)
                failed += 1
                continue

            stereo, metadata = result
            sf.write(str(output_dir / f"{conv_id}.wav"), stereo, cfg["sample_rate"])
            save_json(metadata, meta_path)
            release_claim(claim_path)
            total += 1

    print(f"\nAssembled {total} conversations ({failed} failed/skipped)")


if __name__ == "__main__":
    main()
