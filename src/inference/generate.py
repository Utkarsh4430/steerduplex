"""Offline inference for SteerDuplex.

Uses custom LMGen (ported from PersonaPlex) with:
  - Native 4-phase prompt injection via `provided` tensor
  - Dual Mimi (streaming state isolation)
  - Correct text sampling (temp_text=0.7, top_k_text=25)
  - Voice prompt LUFS normalization (-24 LUFS)
  - Voice prompt embedding save/replay
  - Greedy decoding support
  - Reproducibility seeding

Usage:
    python -m inference.generate --user_audio input.wav --output output.wav
    python -m inference.generate --user_audio input.wav --output output.wav \
        --checkpoint runs/full_v3_.../checkpoint_005000/consolidated
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import sphn

# Disable Triton compilation
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# Add libcuda path
_lib_path = os.environ.get("LIBRARY_PATH", "")
if "/usr/lib/x86_64-linux-gnu" not in _lib_path:
    os.environ["LIBRARY_PATH"] = f"/usr/lib/x86_64-linux-gnu:{_lib_path}".rstrip(":")

from moshi.models import loaders
from safetensors.torch import load_file
from inference.personaplex_loader import is_personaplex, load_personaplex_lm

logger = logging.getLogger(__name__)

N_MIMI_CODEBOOKS = 8


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


class MoshiInference:
    """Moshi model wrapper with PersonaPlex-compatible inference."""

    def __init__(
        self,
        hf_repo_id: str = "kyutai/moshiko-pytorch-bf16",
        checkpoint_path: str | None = None,
        device: str = "cuda",
        seed: int | None = None,
        greedy: bool = False,
        save_voice_prompt_embeddings: bool = False,
    ):
        self.device = torch.device(device)
        self.hf_repo_id = hf_repo_id

        # Set default CUDA device — CUDA graphs require capture on the
        # correct device. This persists for all subsequent operations.
        if self.device.type == "cuda":
            cuda_idx = self.device.index if self.device.index is not None else 0
            torch.cuda.set_device(cuda_idx)

        # Reproducibility
        if seed is not None and seed >= 0:
            from inference.lm_gen import seed_all
            seed_all(seed)

        # CUDA graphs require correct device context during capture.
        # Wrapping all init + warmup in torch.cuda.device() ensures
        # graph capture happens on the right GPU.
        with torch.cuda.device(self.device):
            logger.info("Loading Moshi from %s", hf_repo_id)
            self.checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo=hf_repo_id)

            # Dual Mimi
            # checkpoint_info.get_mimi() computes num_codebooks from lm_config["dep_q"] /
            # lm_config["n_q"], but PersonaPlex's config.json only has {"model_type":...,
            # "version":...} so those keys are absent. Call loaders.get_mimi() directly
            # (defaults to num_codebooks=8, correct for both models).
            logger.info("Loading Mimi codec (x2)...")
            if is_personaplex(self.hf_repo_id):
                self.mimi = loaders.get_mimi(self.checkpoint_info.mimi_weights, device=device)
                self.other_mimi = loaders.get_mimi(self.checkpoint_info.mimi_weights, device=device)
            else:
                self.mimi = self.checkpoint_info.get_mimi(device=device)
                self.other_mimi = self.checkpoint_info.get_mimi(device=device)
            self.mimi.eval()
            self.other_mimi.eval()
            self.sample_rate = self.mimi.sample_rate

            # Text tokenizer
            self.spm = self.checkpoint_info.get_text_tokenizer()

            # LM
            logger.info("Loading Moshi LM...")
            self.lm_model = self._load_lm(checkpoint_path, str(device))
            self.lm_model.eval()
            logger.info("Model loaded (dep_q=%d, n_q=%d).", self.lm_model.dep_q, self.lm_model.n_q)

            # Build LMGen
            from inference.lm_gen import LMGen
            self.lm_gen = LMGen(
                self.lm_model,
                device=device,
                use_sampling=not greedy,
                temp=0.8,
                temp_text=0.7,
                top_k=250,
                top_k_text=25,
                audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                sample_rate=self.sample_rate,
                frame_rate=self.mimi.frame_rate,
                save_voice_prompt_embeddings=save_voice_prompt_embeddings,
            )

            # Enter persistent streaming + warmup (CUDA graphs captured here)
            self.mimi.streaming_forever(1)
            self.other_mimi.streaming_forever(1)
            self.lm_gen.streaming_forever(1)

            self._warmup()

    def _load_lm(self, checkpoint_path: str | None, device: str):
        if is_personaplex(self.hf_repo_id):
            # PersonaPlex config.json only has {"model_type": ..., "version": ...}
            # and cannot be used as LMModel kwargs. Use dedicated loader with
            # correct architecture constants (n_q=16, dep_q=16) and weight patching.
            model = load_personaplex_lm(
                hf_repo=self.hf_repo_id,
                device=device,
                dtype=torch.bfloat16,
            )
            if checkpoint_path:
                self._load_checkpoint_weights(model, checkpoint_path, device)
            return model

        lm_config = dict(
            loaders._lm_kwargs
            if self.checkpoint_info.raw_config is None
            else self.checkpoint_info.raw_config
        )
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

        model = self.checkpoint_info.get_moshi(device=device, lm_kwargs_overrides=lm_config)

        if checkpoint_path:
            self._load_checkpoint_weights(model, checkpoint_path, device)

        return model

    def _load_checkpoint_weights(self, model, checkpoint_path: str, device: str):
        ckpt_dir = Path(checkpoint_path)
        safetensor_files = list(ckpt_dir.glob("*.safetensors"))
        if safetensor_files:
            logger.info("Loading checkpoint from %s", safetensor_files[0])
            state_dict = load_file(str(safetensor_files[0]), device=device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.info("Missing keys: %d", len(missing))
        else:
            pt_files = list(ckpt_dir.glob("*.pt"))
            if pt_files:
                logger.info("Loading checkpoint from %s", pt_files[0])
                state_dict = torch.load(str(pt_files[0]), map_location=device)
                model.load_state_dict(state_dict, strict=False)
            else:
                logger.warning("No checkpoint files found in %s", ckpt_dir)

    def _warmup(self):
        """Run warmup iterations to prime JIT and CUDA graphs."""
        from inference.lm_gen import encode_from_sphn, _iterate_audio
        frame_size = int(self.sample_rate / self.mimi.frame_rate)

        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c:c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:1 + N_MIMI_CODEBOOKS])
                _ = self.other_mimi.decode(tokens[:, 1:1 + N_MIMI_CODEBOOKS])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        logger.info("Warmup complete.")

    @torch.no_grad()
    def generate(
        self,
        user_audio_path: str,
        system_prompt: str = "",
        voice_prompt_path: str | None = None,
        max_duration_sec: float = 30.0,
        temperature: float = 0.8,
        top_k: int = 250,
    ) -> tuple[np.ndarray, int, list[str]]:
        """Generate assistant response.

        Returns:
            Tuple of (audio_array, sample_rate, text_tokens).
        """
        from inference.lm_gen import encode_from_sphn, _iterate_audio

        lm_gen = self.lm_gen
        frame_size = int(self.sample_rate / self.mimi.frame_rate)

        # Reset streaming
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        lm_gen.reset_streaming()

        # Configure prompts
        if system_prompt:
            lm_gen.text_prompt_tokens = self.spm.encode(wrap_with_system_tags(system_prompt))
        else:
            lm_gen.text_prompt_tokens = None

        if voice_prompt_path:
            p = Path(voice_prompt_path)
            if p.suffix == ".pt" and p.exists():
                lm_gen.load_voice_prompt_embeddings(str(p))
            elif p.exists():
                lm_gen.load_voice_prompt(str(p))
            else:
                lm_gen.voice_prompt_audio = None
        else:
            lm_gen.voice_prompt_audio = None

        # Phase 1-4: prompt injection
        lm_gen.step_system_prompts(self.mimi)
        self.mimi.reset_streaming()

        # Load user audio
        user_audio, sr = sphn.read(user_audio_path)
        user_audio = sphn.resample(user_audio, src_sample_rate=sr, dst_sample_rate=self.sample_rate)
        if user_audio.ndim == 1:
            user_audio = user_audio[np.newaxis, :]
        if user_audio.shape[0] > 1:
            user_audio = user_audio[0:1]

        max_samples = int(max_duration_sec * self.sample_rate)
        if user_audio.shape[-1] > max_samples:
            user_audio = user_audio[:, :max_samples]
        total_target_samples = user_audio.shape[-1]

        # Stream and collect output
        output_frames = []
        text_tokens = []

        for user_encoded in encode_from_sphn(
            self.mimi,
            _iterate_audio(user_audio, sample_interval_size=frame_size, pad=True),
        ):
            for c in range(user_encoded.shape[-1]):
                tokens = lm_gen.step(user_encoded[:, :, c:c + 1])
                if tokens is None:
                    continue
                # Decode agent audio
                agent_codes = tokens[:, 1:1 + N_MIMI_CODEBOOKS]
                pcm = self.mimi.decode(agent_codes)
                _ = self.other_mimi.decode(agent_codes)
                output_frames.append(pcm.detach().cpu().numpy()[0, 0])
                # Collect text token
                text_tok = tokens[0, 0, 0].item()
                text_tokens.append(text_tok)

        if not output_frames:
            logger.warning("No output frames generated")
            return np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate, []

        output_audio = np.concatenate(output_frames, axis=-1)

        # Trim/pad to match input duration
        if output_audio.shape[-1] > total_target_samples:
            output_audio = output_audio[:total_target_samples]
        elif output_audio.shape[-1] < total_target_samples:
            output_audio = np.concatenate([
                output_audio,
                np.zeros(total_target_samples - output_audio.shape[-1], dtype=output_audio.dtype),
            ])

        # Decode text tokens to strings
        text_strs = []
        text_token_map = {0: "[EPAD]", 1: "[BOS]", 2: "[EOS]", 3: "[PAD]"}
        for tok in text_tokens:
            if tok in text_token_map:
                text_strs.append(text_token_map[tok])
            else:
                try:
                    text_strs.append(self.spm.id_to_piece(tok).replace("\u2581", " "))
                except Exception:
                    text_strs.append(f"[{tok}]")

        return output_audio.astype(np.float32), self.sample_rate, text_strs


def main():
    parser = argparse.ArgumentParser(description="SteerDuplex offline inference")
    parser.add_argument("--user_audio", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--output_text", type=str, default=None,
                        help="Save decoded text tokens to JSON")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--voice_prompt", type=str, default=None,
                        help="Voice prompt WAV or .pt embeddings")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Seed for reproducibility (-1 disables)")
    parser.add_argument("--greedy", action="store_true",
                        help="Disable sampling (greedy decoding)")
    parser.add_argument("--save_voice_embeddings", action="store_true",
                        help="Save voice prompt embeddings to .pt for fast reuse")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not Path(args.user_audio).exists():
        print(f"Error: {args.user_audio} not found")
        sys.exit(1)

    model = MoshiInference(
        hf_repo_id=args.hf_repo,
        checkpoint_path=args.checkpoint,
        device=args.device,
        seed=args.seed if args.seed >= 0 else None,
        greedy=args.greedy,
        save_voice_prompt_embeddings=args.save_voice_embeddings,
    )

    audio, sr, text_tokens = model.generate(
        user_audio_path=args.user_audio,
        system_prompt=args.system_prompt,
        voice_prompt_path=args.voice_prompt,
        max_duration_sec=args.max_duration,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    import soundfile as sf
    sf.write(args.output, audio, sr)
    print(f"Output: {args.output} ({len(audio)/sr:.1f}s)")

    if args.output_text:
        with open(args.output_text, "w") as f:
            json.dump(text_tokens, f, ensure_ascii=False)
        print(f"Text: {args.output_text}")


if __name__ == "__main__":
    main()
