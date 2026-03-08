"""Offline inference for SteerDuplex.

Generates assistant audio responses given:
- A system prompt (text)
- An assistant voice (voice prompt WAV or preset name)
- User audio input (WAV file)
- Optional: a finetuned checkpoint (LoRA or full)

Based on the Moshi streaming architecture with system prompt injection
following the PersonaPlex pattern: voice_prompt → silence → text_prompt → silence → conversation.

Usage:
    # Basic inference with base model
    python -m inference.generate \
        --user_audio input.wav \
        --output output.wav \
        --system_prompt "You are a helpful assistant."

    # With finetuned checkpoint
    python -m inference.generate \
        --user_audio input.wav \
        --output output.wav \
        --system_prompt "You are a helpful assistant." \
        --checkpoint runs/pilot_v1/checkpoint_3000

    # With voice prompt
    python -m inference.generate \
        --user_audio input.wav \
        --output output.wav \
        --system_prompt "You are a helpful assistant." \
        --voice_prompt data/voices/assistant/audio/ryan_ref.wav

    # Interactive mode (microphone)
    python -m inference.generate --interactive \
        --system_prompt "You are a friendly tutor." \
        --voice_prompt data/voices/assistant/audio/vivian_ref.wav
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import sphn

from moshi.models import loaders

logger = logging.getLogger(__name__)


def wrap_with_system_tags(text: str) -> str:
    """Wrap text with <system> tags for Moshi conditioning."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    """Load audio file and return as tensor (1, T)."""
    audio, sr = sphn.read(path)
    audio = torch.from_numpy(audio)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio[0:1]  # mono
    if sr != sample_rate:
        import torchaudio.functional as F
        audio = F.resample(audio, sr, sample_rate)
    return audio


def normalize_audio(audio: torch.Tensor, sample_rate: int, target_lufs: float = -24.0) -> torch.Tensor:
    """Normalize audio to target LUFS (simplified loudness normalization)."""
    rms = audio.pow(2).mean().sqrt()
    if rms > 0:
        # Approximate LUFS normalization via RMS scaling
        target_rms = 10 ** (target_lufs / 20)
        audio = audio * (target_rms / rms)
    return audio


class MoshiInference:
    """Moshi model wrapper for offline inference with system prompt support."""

    def __init__(
        self,
        hf_repo_id: str = "kyutai/moshiko-pytorch-bf16",
        checkpoint_path: str | None = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)

        # Load model components
        logger.info("Loading Moshi from %s", hf_repo_id)
        self.checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo=hf_repo_id,
        )

        # Load Mimi codec
        logger.info("Loading Mimi codec...")
        self.mimi = self.checkpoint_info.get_mimi(device=device)
        self.mimi.eval()
        self.sample_rate = self.mimi.sample_rate

        # Load text tokenizer
        self.spm = self.checkpoint_info.get_text_tokenizer()

        # Load LM
        logger.info("Loading Moshi LM...")
        lm_config = (
            loaders._lm_kwargs
            if self.checkpoint_info.raw_config is None
            else self.checkpoint_info.raw_config
        )

        # If checkpoint has LoRA, enable it
        if checkpoint_path:
            ckpt_dir = Path(checkpoint_path)
            config_path = ckpt_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    ckpt_config = json.load(f)
                if ckpt_config.get("lora"):
                    lm_config["lora"] = True
                    lm_config["lora_rank"] = ckpt_config.get("lora_rank", 128)
                    lm_config["lora_scaling"] = ckpt_config.get("lora_scaling", 2.0)

        self.lm_model = self.checkpoint_info.get_moshi(device=device, config=lm_config)

        # Load finetuned weights if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self.lm_model.eval()
        logger.info("Model loaded successfully.")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load finetuned weights (LoRA or full) from checkpoint directory."""
        ckpt_dir = Path(checkpoint_path)

        # Look for safetensors or .pt checkpoint files
        from safetensors.torch import load_file

        safetensor_files = list(ckpt_dir.glob("*.safetensors"))
        if safetensor_files:
            logger.info("Loading checkpoint from %s", safetensor_files[0])
            state_dict = load_file(str(safetensor_files[0]))
            missing, unexpected = self.lm_model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("Missing keys: %d (expected for LoRA)", len(missing))
            if unexpected:
                logger.warning("Unexpected keys: %d", len(unexpected))
        else:
            pt_files = list(ckpt_dir.glob("*.pt"))
            if pt_files:
                logger.info("Loading checkpoint from %s", pt_files[0])
                state_dict = torch.load(str(pt_files[0]), map_location=self.device)
                self.lm_model.load_state_dict(state_dict, strict=False)
            else:
                logger.warning("No checkpoint files found in %s", ckpt_dir)

    @torch.no_grad()
    def generate(
        self,
        user_audio_path: str,
        system_prompt: str = "",
        voice_prompt_path: str | None = None,
        max_duration_sec: float = 30.0,
        temperature: float = 0.8,
        top_k: int = 250,
    ) -> tuple[np.ndarray, int]:
        """Generate assistant response for given user audio.

        Args:
            user_audio_path: Path to user audio WAV file.
            system_prompt: System prompt text for conditioning.
            voice_prompt_path: Optional path to voice prompt WAV (3-10s).
            max_duration_sec: Maximum output duration in seconds.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        from moshi.models.lm import LMGen

        # Tokenize system prompt
        text_prompt_tokens = None
        if system_prompt:
            tagged_prompt = wrap_with_system_tags(system_prompt)
            text_prompt_tokens = torch.tensor(
                self.spm.encode(tagged_prompt),
                dtype=torch.long,
                device=self.device,
            )

        # Build LMGen
        lm_gen = LMGen(
            self.lm_model,
            temp=temperature,
            top_k=top_k,
            text_prompt_tokens=text_prompt_tokens,
        )

        # Load voice prompt if provided
        if voice_prompt_path and Path(voice_prompt_path).exists():
            lm_gen.load_voice_prompt(voice_prompt_path)

        # Reset streaming state
        self.mimi.reset_streaming()
        lm_gen.reset_streaming()

        # Step through system prompts (voice → silence → text → silence)
        lm_gen.step_system_prompts(self.mimi)

        # Load and encode user audio
        user_audio = load_audio(user_audio_path, self.sample_rate)
        user_audio = user_audio.to(self.device)

        # Limit duration
        max_samples = int(max_duration_sec * self.sample_rate)
        if user_audio.shape[-1] > max_samples:
            user_audio = user_audio[..., :max_samples]

        # Encode user audio frame by frame with Mimi
        frame_size = int(self.sample_rate / self.mimi.frame_rate)
        num_frames = user_audio.shape[-1] // frame_size

        output_frames = []

        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = user_audio[..., start:end]

            if frame.ndim == 2:
                frame = frame.unsqueeze(0)  # (1, 1, frame_size)

            # Encode user frame
            user_codes = self.mimi.encode(frame)

            # Step the LM: feed user codes, get assistant codes
            out = lm_gen.step(input_tokens=user_codes)

            if out is not None:
                # Decode assistant codes to audio
                assistant_audio = self.mimi.decode(out.unsqueeze(0))
                output_frames.append(assistant_audio.squeeze().cpu())

        if not output_frames:
            logger.warning("No output frames generated")
            return np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate

        # Concatenate output
        output_audio = torch.cat(output_frames, dim=-1).numpy()

        # Normalize output
        peak = np.abs(output_audio).max()
        if peak > 0:
            output_audio = output_audio * (0.95 / peak)

        return output_audio.astype(np.float32), self.sample_rate


def main():
    parser = argparse.ArgumentParser(
        description="SteerDuplex offline inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python -m inference.generate --user_audio input.wav --output output.wav

  # With system prompt and voice
  python -m inference.generate \\
      --user_audio input.wav \\
      --output output.wav \\
      --system_prompt "You are a friendly tutor." \\
      --voice_prompt voices/ryan_ref.wav

  # With finetuned model
  python -m inference.generate \\
      --user_audio input.wav \\
      --output output.wav \\
      --checkpoint runs/pilot_v1/checkpoint_3000 \\
      --system_prompt "You are a helpful assistant."
        """,
    )

    parser.add_argument("--user_audio", type=str, required=True, help="Path to user audio WAV file")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV path")
    parser.add_argument("--system_prompt", type=str, default="", help="System prompt text")
    parser.add_argument("--voice_prompt", type=str, default=None, help="Path to voice prompt WAV (3-10s)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to finetuned checkpoint dir")
    parser.add_argument("--hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16", help="HuggingFace model repo")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max output duration (seconds)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=250, help="Top-k sampling")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not Path(args.user_audio).exists():
        print(f"Error: user audio not found: {args.user_audio}")
        sys.exit(1)

    # Load model
    model = MoshiInference(
        hf_repo_id=args.hf_repo,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Generate
    print(f"Generating response for: {args.user_audio}")
    if args.system_prompt:
        print(f"System prompt: {args.system_prompt}")
    if args.voice_prompt:
        print(f"Voice prompt: {args.voice_prompt}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")

    audio, sr = model.generate(
        user_audio_path=args.user_audio,
        system_prompt=args.system_prompt,
        voice_prompt_path=args.voice_prompt,
        max_duration_sec=args.max_duration,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # Save output
    import soundfile as sf
    sf.write(args.output, audio, sr)
    print(f"Output saved to: {args.output} ({len(audio)/sr:.1f}s)")


if __name__ == "__main__":
    main()
