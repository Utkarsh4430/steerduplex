"""Phase 3: Synthesize speech using Qwen3-TTS.

Assistant turns: CustomVoice model (preset speaker + instruct for style control).
User turns: Base model (voice cloning via ref_audio, no style control).

Resumable: skips already-synthesized turns.

Usage:
    python -m pipeline.synthesize_tts --config configs/generation.yaml
    python -m pipeline.synthesize_tts --config configs/generation.yaml --category A3_tone_controlled
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from pipeline.utils import ensure_dir, load_json, load_yaml, save_json, set_seed


class TTSSynthesizer:
    """Dual-model Qwen3-TTS wrapper.

    Loads two models:
    - CustomVoice: for assistant turns (preset speaker + instruct)
    - Base: for user turns (voice cloning from ref_audio)
    """

    def __init__(
        self,
        assistant_model_id: str,
        user_model_id: str,
        device: str = "cuda:0",
    ):
        from qwen_tts import Qwen3TTSModel

        print(f"Loading CustomVoice model: {assistant_model_id}")
        self.cv_model = Qwen3TTSModel.from_pretrained(
            assistant_model_id,
            device_map=device,
            dtype=torch.bfloat16,
        )

        print(f"Loading Base model: {user_model_id}")
        self.base_model = Qwen3TTSModel.from_pretrained(
            user_model_id,
            device_map=device,
            dtype=torch.bfloat16,
        )

        self._user_prompts: dict[str, object] = {}  # cache voice clone prompts
        print("TTS models loaded.")

    def synthesize_assistant(
        self,
        text: str,
        speaker: str,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize assistant turn using CustomVoice (preset + instruct).

        Returns (waveform, sample_rate).
        """
        kwargs = dict(
            text=text,
            language="English",
            speaker=speaker,
        )
        if instruct:
            kwargs["instruct"] = instruct

        wavs, sr = self.cv_model.generate_custom_voice(**kwargs)
        audio = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio.astype(np.float32), sr

    def synthesize_user_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: str = "",
    ) -> tuple[np.ndarray, int]:
        """Synthesize user turn using Base model (voice cloning).

        Returns (waveform, sample_rate).
        """
        # Cache the voice clone prompt for reuse across turns
        if ref_audio not in self._user_prompts:
            self._user_prompts[ref_audio] = self.base_model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text or "Reference audio.",
                x_vector_only_mode=not bool(ref_text),
            )

        wavs, sr = self.base_model.generate_voice_clone(
            text=text,
            language="English",
            voice_clone_prompt=self._user_prompts[ref_audio],
        )
        audio = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio.astype(np.float32), sr

    def synthesize_user_preset(
        self,
        text: str,
        speaker: str,
    ) -> tuple[np.ndarray, int]:
        """Fallback: synthesize user turn using CustomVoice preset (no cloning)."""
        wavs, sr = self.cv_model.generate_custom_voice(
            text=text,
            language="English",
            speaker=speaker,
        )
        audio = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio.astype(np.float32), sr


def resample_if_needed(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio if sample rates don't match."""
    if src_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=src_sr, target_sr=target_sr)


def synthesize_conversation(
    synth: TTSSynthesizer,
    transcript: dict,
    output_dir: Path,
    target_sr: int = 24000,
    max_retries: int = 3,
) -> dict | None:
    """Synthesize all turns of a conversation. Saves each turn immediately.

    Returns updated transcript with audio_path per turn, or None on failure.
    """
    conv_id = transcript["id"]
    conv_dir = ensure_dir(output_dir / conv_id)

    assistant_voice = transcript.get("assistant_voice", {})
    user_voice = transcript.get("user_voice", {})

    updated_turns = []
    for i, turn in enumerate(transcript["turns"]):
        role = turn["role"]
        text = turn["text"]
        out_path = conv_dir / f"turn_{i:03d}_{role}.wav"

        # Resume: skip if already exists
        if out_path.exists():
            updated_turns.append({**turn, "audio_path": str(out_path)})
            continue

        success = False
        for retry in range(max_retries):
            try:
                if role == "assistant":
                    audio, sr = synth.synthesize_assistant(
                        text=text,
                        speaker=assistant_voice.get("speaker", "Ryan"),
                        instruct=turn.get("tts_instruct"),
                    )
                else:
                    # User: clone or preset fallback
                    if user_voice.get("model") == "Base" and user_voice.get("ref_path"):
                        audio, sr = synth.synthesize_user_clone(
                            text=text,
                            ref_audio=user_voice["ref_path"],
                            ref_text=user_voice.get("ref_text", ""),
                        )
                    else:
                        audio, sr = synth.synthesize_user_preset(
                            text=text,
                            speaker=user_voice.get("speaker", "Eric"),
                        )

                # Resample to target and save
                audio = resample_if_needed(audio, sr, target_sr)
                # Normalize
                peak = np.abs(audio).max()
                if peak > 0:
                    audio = audio * (0.95 / peak)
                sf.write(str(out_path), audio, target_sr)

                updated_turns.append({**turn, "audio_path": str(out_path)})
                success = True
                break

            except Exception as e:
                print(f"  [RETRY {retry + 1}] {conv_id} turn {i} ({role}): {e}")

        if not success:
            print(f"  [FAIL] {conv_id} turn {i}")
            return None

    result = {**transcript, "turns": updated_turns}
    return result


def main():
    parser = argparse.ArgumentParser(description="Synthesize TTS audio")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--assignments_dir", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["tts"]
    set_seed(args.seed)

    synth = TTSSynthesizer(
        assistant_model_id=cfg["assistant_model_id"],
        user_model_id=cfg["user_model_id"],
        device=cfg.get("device", "cuda:0"),
    )

    assignments_dir = Path(args.assignments_dir or load_yaml(args.config)["voices"]["output_dir"])
    output_dir = ensure_dir(cfg["output_dir"])
    target_sr = cfg.get("sample_rate", 24000)

    for cat_dir in sorted(assignments_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.category and cat_dir.name != args.category:
            continue

        cat_output = ensure_dir(output_dir / cat_dir.name)
        transcripts = sorted(cat_dir.glob("*.json"))

        # Resume: find already completed
        done = {p.stem.replace("_synth", "") for p in cat_output.glob("*_synth.json")}
        remaining = [t for t in transcripts if t.stem not in done]

        if not remaining:
            print(f"[SKIP] {cat_dir.name}: all {len(transcripts)} done")
            continue

        print(f"\n=== {cat_dir.name}: {len(remaining)} remaining ({len(done)} done) ===")

        for transcript_path in tqdm(remaining, desc=cat_dir.name):
            transcript = load_json(transcript_path)
            result = synthesize_conversation(
                synth, transcript, cat_output,
                target_sr=target_sr,
                max_retries=cfg.get("max_retries", 3),
            )
            if result:
                save_json(result, cat_output / f"{transcript['id']}_synth.json")


if __name__ == "__main__":
    main()
