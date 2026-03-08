"""Phase 3: Synthesize speech using Qwen3-TTS.

Generates per-turn audio files from transcripts with voice cloning
and style control via the `instruct` parameter.

Usage:
    python -m pipeline.synthesize_tts \
        --config configs/generation.yaml \
        --assignments_dir data/voice_assignments \
        --category A3_tone_controlled
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from pipeline.utils import ensure_dir, load_json, load_yaml, save_json, set_seed


class TTSSynthesizer:
    """Wrapper around Qwen3-TTS for voice cloning + style control."""

    def __init__(self, model_id: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Qwen3-TTS from {model_id}...")
        self.device = device

        # Qwen3-TTS uses a custom pipeline - adjust based on actual API
        # This is the expected interface based on Qwen3-TTS docs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.sample_rate = 24000
        print("TTS model loaded.")

    def synthesize(
        self,
        text: str,
        ref_audio_path: str | None = None,
        instruct: str | None = None,
    ) -> np.ndarray:
        """Generate speech from text with optional voice cloning and style control.

        Args:
            text: Text to speak
            ref_audio_path: Path to reference audio for voice cloning
            instruct: Style instruction (e.g. "sarcastic, dry delivery")

        Returns:
            Audio waveform as numpy array (mono, self.sample_rate Hz)
        """
        # Build the generation prompt based on Qwen3-TTS API
        # The exact API depends on Qwen3-TTS version; adjust as needed
        messages = []

        if instruct:
            messages.append({"role": "system", "content": f"[instruct] {instruct}"})

        content_parts = []
        if ref_audio_path and Path(ref_audio_path).exists():
            content_parts.append({"type": "audio", "audio": ref_audio_path})
        content_parts.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content_parts})

        # Generate using Qwen3-TTS
        # NOTE: The actual Qwen3-TTS API may differ. This matches the expected
        # transformers-based interface. Adjust after testing.
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", tokenize=True
        )
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs if isinstance(inputs, dict) else {"input_ids": inputs},
                max_new_tokens=4096,
            )

        # Decode audio from model output
        # Qwen3-TTS outputs audio tokens that need decoding
        audio = self._decode_audio(outputs)
        return audio

    def _decode_audio(self, outputs) -> np.ndarray:
        """Decode model outputs to audio waveform.

        NOTE: Implementation depends on Qwen3-TTS audio codec.
        This is a placeholder that should be adapted once the
        actual Qwen3-TTS decoding pipeline is confirmed.
        """
        # Qwen3-TTS typically has a dedicated decode method
        if hasattr(self.model, "decode_audio"):
            audio = self.model.decode_audio(outputs)
        elif hasattr(self.model, "generate_audio"):
            audio = outputs  # Some models return audio directly
        else:
            # Fallback: assume outputs contain audio tokens to decode
            audio = outputs.cpu().numpy().flatten()

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy().flatten()

        # Normalize
        if audio.max() > 0:
            audio = audio / max(abs(audio.max()), abs(audio.min())) * 0.95

        return audio.astype(np.float32)


def synthesize_conversation(
    synthesizer: TTSSynthesizer,
    transcript: dict,
    output_dir: Path,
    max_retries: int = 3,
) -> dict | None:
    """Synthesize all turns of a conversation.

    Returns updated transcript with audio paths, or None on failure.
    """
    conv_id = transcript["id"]
    conv_dir = ensure_dir(output_dir / conv_id)
    assistant_ref = transcript.get("assistant_voice", {}).get("ref_path")
    user_ref = transcript.get("user_voice", {}).get("ref_path")

    audio_turns = []
    for i, turn in enumerate(transcript["turns"]):
        role = turn["role"]
        text = turn["text"]
        tts_instruct = turn.get("tts_instruct")
        ref_path = assistant_ref if role == "assistant" else user_ref

        out_path = conv_dir / f"turn_{i:03d}_{role}.wav"

        if out_path.exists():
            audio_turns.append(
                {
                    **turn,
                    "audio_path": str(out_path),
                }
            )
            continue

        success = False
        for retry in range(max_retries):
            try:
                audio = synthesizer.synthesize(
                    text=text,
                    ref_audio_path=ref_path,
                    instruct=tts_instruct,
                )
                sf.write(str(out_path), audio, synthesizer.sample_rate)
                audio_turns.append(
                    {
                        **turn,
                        "audio_path": str(out_path),
                    }
                )
                success = True
                break
            except Exception as e:
                print(f"  [RETRY {retry+1}] Turn {i} ({role}): {e}")

        if not success:
            print(f"  [FAIL] Could not synthesize turn {i} for {conv_id}")
            return None

    transcript["turns"] = audio_turns
    return transcript


def main():
    parser = argparse.ArgumentParser(description="Synthesize TTS audio")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--assignments_dir", type=str, default="data/voice_assignments")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["tts"]
    set_seed(args.seed)

    synthesizer = TTSSynthesizer(model_id=cfg["model_id"], device=cfg["device"])
    output_dir = ensure_dir(cfg["output_dir"])
    assignments_dir = Path(args.assignments_dir)

    for cat_dir in sorted(assignments_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.category and cat_dir.name != args.category:
            continue

        cat_output = ensure_dir(output_dir / cat_dir.name)
        transcripts = sorted(cat_dir.glob("*.json"))
        print(f"\n=== Synthesizing {len(transcripts)} conversations for {cat_dir.name} ===")

        for transcript_path in tqdm(transcripts, desc=cat_dir.name):
            transcript = load_json(transcript_path)
            result = synthesize_conversation(
                synthesizer, transcript, cat_output, max_retries=cfg["max_retries"]
            )
            if result:
                save_json(result, cat_output / f"{transcript['id']}_synth.json")


if __name__ == "__main__":
    main()
