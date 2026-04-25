"""Custom LMGen for SteerDuplex inference.

Ported from PersonaPlex (NVIDIA, MIT License) with adaptations for the
upstream Kyutai moshi package. Features:

  - 3-argument step(): input_tokens (user), moshi_tokens (agent), text_token
  - Explicit `provided` tensor for force-feeding during prompt injection
  - Native step_system_prompts() — no cache manipulation
  - Voice prompt embedding save/replay (.pt cache)
  - LUFS normalization for voice prompts (-24 LUFS via pyloudnorm)
  - Greedy decoding support

Based on: https://github.com/NVIDIA/personaplex/blob/main/moshi/moshi/models/lm.py
"""

from dataclasses import dataclass
from os.path import splitext
from typing import Optional

import numpy as np
import torch
import sphn

from moshi.modules.streaming import StreamingModule

try:
    from moshi.utils.compile import CUDAGraphed
except ImportError:
    from moshi.modules.streaming import CUDAGraphed

try:
    from moshi.utils.sampling import sample_token
except ImportError:
    def sample_token(logits, use_sampling=True, temp=1.0, top_k=0, top_p=0.0):
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_k > 0:
                probs, indices = torch.topk(probs, top_k, dim=-1)
                q = torch.empty_like(probs).exponential_(1)
                next_token = indices.gather(-1, (probs / q).argmax(-1, keepdim=True))
            else:
                q = torch.empty_like(probs).exponential_(1)
                next_token = (probs / q).argmax(-1, keepdim=True)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token[..., 0]


# ---------------------------------------------------------------------------
# Constants (matching PersonaPlex exactly)
# ---------------------------------------------------------------------------
AUDIO_TOKENS_PER_STREAM = 8
FRAME_RATE_HZ = 12.5
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def normalize_audio(wav: np.ndarray, sr: int, target_lufs: float = -24.0) -> np.ndarray:
    """Normalize mono audio to target LUFS level using pyloudnorm."""
    try:
        import pyloudnorm as pyln
        if wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav[0]
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        if np.isinf(loudness):
            return wav  # silence, can't normalize
        return pyln.normalize.loudness(wav, loudness, target_lufs)
    except Exception:
        # Fallback: RMS normalization
        rms = np.sqrt(np.mean(wav ** 2))
        if rms > 0:
            target_rms = 10 ** (target_lufs / 20)
            wav = wav * (target_rms / rms)
        return wav


def _iterate_audio(sample_pcm, sample_interval_size, max_len=None, pad=True):
    """Yield audio in chunks of sample_interval_size."""
    import sys
    if max_len is None:
        max_len = sys.maxsize
    cnt = 0
    while sample_pcm.shape[-1] > 0 and cnt < max_len:
        sample = sample_pcm[:, :sample_interval_size]
        sample_pcm = sample_pcm[:, sample_interval_size:]
        if sample_pcm.shape[-1] == 0 and pad and sample.shape[-1] < sample_interval_size:
            sample = np.concatenate([
                sample,
                np.zeros((sample.shape[0], sample_interval_size - sample.shape[-1])),
            ], axis=1)
        cnt += 1
        yield sample[0:1]  # (1, T)


def encode_from_sphn(mimi, samples, max_batch=1):
    """Encode audio samples through Mimi, yielding codec frames."""
    device = next(mimi.parameters()).device
    current_batch = []
    done_flag = False

    while True:
        try:
            sample = next(samples)
            tensor = torch.tensor(sample, dtype=torch.float32, device=device)
            tensor = tensor.unsqueeze(0)  # (1, C, T)
            current_batch.append(tensor)
        except StopIteration:
            done_flag = True

        if not done_flag and len(current_batch) < max_batch:
            continue
        if not current_batch:
            break

        batch = torch.cat(current_batch, dim=0)
        encoded = mimi.encode(batch)
        for x in torch.unbind(encoded, dim=0):
            yield x.unsqueeze(0).detach().clone()
        current_batch = []

        if done_flag:
            break


def seed_all(seed: int):
    """Seed torch, CUDA, numpy, and Python RNG for reproducible runs."""
    import random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class _LMGenState:
    batch_size: int
    device: torch.device
    cache: torch.Tensor       # [B, num_codebooks, max_delay+3]
    provided: torch.Tensor    # [B, num_codebooks, max_delay+3] bool
    initial: torch.Tensor     # [B, num_codebooks, 1]
    graphed_main: CUDAGraphed
    graphed_embeddings: CUDAGraphed
    graphed_depth: CUDAGraphed
    offset: int = 0

    def __post_init__(self):
        self.exec_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    def reset(self, reset_mask: torch.Tensor | None = None):
        self.offset = 0
        self.provided[:] = False
        if reset_mask is not None:
            self.exec_mask[:] = torch.where(
                reset_mask, torch.ones_like(self.exec_mask), self.exec_mask,
            )

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


# ---------------------------------------------------------------------------
# LMGen
# ---------------------------------------------------------------------------
class LMGen(StreamingModule[_LMGenState]):
    """PersonaPlex-style LMGen with native prompt injection support."""

    def __init__(
        self,
        lm_model,
        device: str | torch.device = "cuda",
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        audio_silence_frame_cnt: int = 6,
        text_prompt_tokens: list[int] | None = None,
        save_voice_prompt_embeddings: bool = False,
        sample_rate: int = 24000,
        frame_rate: float = FRAME_RATE_HZ,
    ):
        assert not lm_model.training, "LMGen should not be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.text_prompt_tokens = text_prompt_tokens
        self.audio_silence_frame_cnt = audio_silence_frame_cnt
        self.zero_text_code = 3  # PAD token
        self.save_voice_prompt_embeddings = save_voice_prompt_embeddings

        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._frame_size = int(sample_rate / frame_rate)

        self.max_delay = max(lm_model.delays)
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long,
        )

        # Precomputed frames (avoid repeated tensor creation)
        self._silence_frame = torch.as_tensor(
            SILENCE_TOKENS, dtype=torch.long, device=lm_model.device,
        ).view(1, AUDIO_TOKENS_PER_STREAM, 1)
        self._sine_frame = torch.as_tensor(
            SINE_TOKENS, dtype=torch.long, device=lm_model.device,
        ).view(1, AUDIO_TOKENS_PER_STREAM, 1)

        # Voice prompt state
        self.voice_prompt: str | None = None
        self.voice_prompt_audio: np.ndarray | None = None
        self.voice_prompt_embeddings: torch.Tensor | None = None
        self.voice_prompt_cache: torch.Tensor | None = None

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        lm_model = self.lm_model
        device = lm_model.device
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, lm_model.num_codebooks, self.max_delay + 3),
            lm_model.ungenerated_token_id, device=device, dtype=torch.long,
        )
        provided = torch.full(
            (batch_size, lm_model.num_codebooks, self.max_delay + 3),
            False, device=device, dtype=torch.bool,
        )

        disable = device.type != "cuda"
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=disable)
        graphed_embeddings = CUDAGraphed(self._forward_embeddings, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        return _LMGenState(
            batch_size=batch_size, device=device, cache=cache,
            provided=provided, initial=initial,
            graphed_main=graphed_main,
            graphed_embeddings=graphed_embeddings,
            graphed_depth=graphed_depth,
        )

    # ------------------------------------------------------------------
    # Token encoding helpers (precomputed, no allocation per call)
    # ------------------------------------------------------------------
    def _encode_silence_frame(self) -> torch.Tensor:
        return self._silence_frame

    def _encode_sine_frame(self) -> torch.Tensor:
        return self._sine_frame

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prepare_step_input(
        self,
        input_tokens: torch.Tensor | None = None,
        moshi_tokens: torch.Tensor | None = None,
        text_token: int | torch.Tensor | None = None,
    ):
        state = self._streaming_state
        if state is None:
            raise RuntimeError("Must be in streaming mode.")
        lm_model = self.lm_model

        needed_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        CT = state.cache.shape[2]

        if input_tokens is not None:
            assert input_tokens.dim() == 3
            B, Ki, S = input_tokens.shape
            assert S == 1 and Ki == needed_tokens
            for q in range(Ki):
                k = AUDIO_TOKENS_PER_STREAM + 1 + q
                delay = lm_model.delays[k]
                pos = (state.offset + delay) % CT
                state.cache[:, k, pos:pos + 1] = input_tokens[:, q]
                state.provided[:, k, pos:pos + 1] = True

        if moshi_tokens is not None:
            assert moshi_tokens.dim() == 3
            B, Ki, S = moshi_tokens.shape
            assert S == 1 and Ki == needed_tokens
            for q in range(Ki):
                k = 1 + q
                delay = lm_model.delays[k]
                pos = (state.offset + delay) % CT
                state.cache[:, k, pos:pos + 1] = moshi_tokens[:, q]
                state.provided[:, k, pos:pos + 1] = True

        if text_token is not None:
            pos = (state.offset + lm_model.delays[0]) % CT
            state.cache[:, 0, pos] = (
                text_token if isinstance(text_token, int)
                else text_token.item() if text_token.dim() == 0
                else text_token
            )
            state.provided[:, 0, pos] = True

        for k, delay in enumerate(lm_model.delays):
            if state.offset <= delay:
                state.cache[:, k, state.offset % CT] = state.initial[:, k, 0]
                state.provided[:, k, state.offset % CT] = True

        if state.offset == 0:
            state.cache[:, :, 0] = state.initial[:, :, 0]
            state.offset += 1
            return None

        model_pos = (state.offset - 1) % CT
        target_pos = state.offset % CT
        input_ = state.cache[:, :, model_pos:model_pos + 1]
        target_ = state.cache[:, :, target_pos:target_pos + 1]
        provided_ = state.provided[:, :, target_pos:target_pos + 1]

        return input_, provided_, target_, model_pos, target_pos

    @torch.no_grad()
    def step(
        self,
        input_tokens: torch.Tensor | None = None,
        moshi_tokens: torch.Tensor | None = None,
        text_token: int | torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | None:
        """Run one step. Returns output tokens [B, dep_q+1, 1] or None."""
        state = self._streaming_state

        prepared = self.prepare_step_input(input_tokens, moshi_tokens, text_token)
        if prepared is None:
            return None

        input_, provided_, target_, model_pos, target_pos = prepared

        transformer_out, text_logits = state.graphed_main(input_)

        return self._process_output(
            transformer_out, text_logits, provided_, target_, model_pos, target_pos,
        )

    def _forward_embeddings(self, input_: torch.Tensor):
        """Run the main transformer + text head on a pre-computed embedding tensor.

        Mirrors PersonaPlex LMModel.forward_embeddings — skips the code→embedding
        step so that saved voice-prompt embeddings can be replayed through the
        transformer (re-establishing its streaming KV state).
        """
        lm = self.lm_model
        transformer_out = lm.transformer(input_)
        if lm.out_norm is not None:
            transformer_out = lm.out_norm(transformer_out)
        text_logits = lm.text_linear(transformer_out)[:, None]
        return transformer_out, text_logits

    @torch.no_grad()
    def step_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor | None:
        """Advance one step by feeding pre-computed embeddings to the transformer.

        Used to replay saved voice-prompt embeddings on the .pt load path.
        Matches PersonaPlex step_embeddings: dummy tokens satisfy the cache-write
        invariants in prepare_step_input, but the transformer is driven by the
        embeddings argument (not a cache read).
        """
        state = self._streaming_state
        lm = self.lm_model
        needed = lm.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        dummy = lm._get_initial_token()
        while True:
            prepared = self.prepare_step_input(
                input_tokens=dummy[:, 1:1 + needed],
                moshi_tokens=dummy[:, 1 + needed:],
                text_token=self.zero_text_code,
            )
            if prepared is not None:
                break
        _, provided_, target_, model_pos, target_pos = prepared
        transformer_out, text_logits = state.graphed_embeddings(embeddings)
        return self._process_output(
            transformer_out, text_logits, provided_, target_, model_pos, target_pos,
        )

    @torch.no_grad()
    def _process_output(
        self, transformer_out, text_logits, provided_, target_, model_pos, target_pos,
    ) -> torch.Tensor | None:
        state = self._streaming_state
        lm_model = self.lm_model

        sampled_text = sample_token(
            text_logits.float(), self.use_sampling, self.temp_text, self.top_k_text,
        )
        assert sampled_text.dim() == 3
        sampled_text = sampled_text[:, 0, 0]

        next_text = torch.where(provided_[:, 0, 0], target_[:, 0, 0], sampled_text)

        sampled_audio = state.graphed_depth(
            next_text, transformer_out,
            target_[:, lm_model.audio_offset:, 0],
            provided_[:, lm_model.audio_offset:, 0],
        )

        state.provided[:, :, model_pos] = False

        state.cache[:, 0, target_pos] = torch.where(
            ~state.provided[:, 0, target_pos],
            sampled_text,
            state.cache[:, 0, target_pos],
        )
        state.cache[:, 1:lm_model.dep_q + 1, target_pos] = torch.where(
            ~state.provided[:, 1:lm_model.dep_q + 1, target_pos],
            sampled_audio,
            state.cache[:, 1:lm_model.dep_q + 1, target_pos],
        )

        if state.offset <= self.max_delay:
            state.offset += 1
            return None

        B = state.cache.shape[0]
        CT = state.cache.shape[2]
        gen_delays = self.delays_cuda[:lm_model.dep_q + 1]
        index = (
            ((state.offset - self.max_delay + gen_delays) % CT)
            .view(1, -1, 1).expand(B, -1, 1)
        )
        out = state.cache.gather(dim=2, index=index)
        state.offset += 1
        return out

    def depformer_step(
        self, text_token, transformer_out, audio_tokens, audio_provided,
    ) -> torch.Tensor:
        (B,) = text_token.shape
        prev_token = text_token
        lm_model = self.lm_model
        depformer_tokens = []

        assert not lm_model.depformer.is_streaming
        with lm_model.depformer.streaming(B):
            for cb_index in range(lm_model.dep_q):
                input_ = prev_token[:, None, None]
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                next_token = sample_token(
                    logits.float(), self.use_sampling, self.temp, self.top_k,
                )
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]
                prev_token = torch.where(
                    audio_provided[:, cb_index],
                    audio_tokens[:, cb_index],
                    next_token,
                )
                depformer_tokens.append(next_token)

        return torch.stack(depformer_tokens, dim=1)

    # ------------------------------------------------------------------
    # Voice prompt (with LUFS normalization + embedding save/replay)
    # ------------------------------------------------------------------
    def load_voice_prompt(self, path: str):
        """Load voice prompt WAV with -24 LUFS normalization."""
        self.voice_prompt = path
        audio, sr = sphn.read(path)
        audio = sphn.resample(audio, src_sample_rate=sr, dst_sample_rate=self._sample_rate)
        # Normalize to -24 LUFS (mono)
        if audio.ndim == 2:
            audio = normalize_audio(audio[0], self._sample_rate, -24.0)
            audio = audio[np.newaxis, :]
        else:
            audio = normalize_audio(audio, self._sample_rate, -24.0)
            audio = audio[np.newaxis, :]
        self.voice_prompt_audio = audio
        self.voice_prompt_embeddings = None
        self.voice_prompt_cache = None

    def load_voice_prompt_embeddings(self, path: str):
        """Load pre-saved voice prompt embeddings (.pt) for fast replay."""
        self.voice_prompt = path
        state = torch.load(path, map_location=self.lm_model.device)
        self.voice_prompt_audio = None
        self.voice_prompt_embeddings = state["embeddings"].to(self.lm_model.device)
        self.voice_prompt_cache = state["cache"].to(self.lm_model.device)

    # ------------------------------------------------------------------
    # System prompt injection (PersonaPlex 4-phase pattern)
    # ------------------------------------------------------------------
    def _step_voice_prompt(self, mimi):
        if self.voice_prompt_audio is None and self.voice_prompt_embeddings is None:
            return

        if self.voice_prompt_embeddings is not None:
            # Replay saved embeddings through the transformer to re-establish its
            # streaming KV state and advance state.offset past the offset==0
            # sentinel branch, then overlay the stored LMGen cache.
            for next_embed in self.voice_prompt_embeddings:
                self.step_embeddings(next_embed)
            self._streaming_state.cache.copy_(self.voice_prompt_cache)
            return

        # Encode voice prompt through Mimi
        saved_embeddings = []
        for frame_codes in encode_from_sphn(
            mimi,
            _iterate_audio(
                self.voice_prompt_audio,
                sample_interval_size=self._frame_size,
                pad=True,
            ),
        ):
            self.step(
                moshi_tokens=frame_codes,
                text_token=self.zero_text_code,
                input_tokens=self._encode_sine_frame(),
            )

        # Save embeddings if requested
        if self.save_voice_prompt_embeddings and self.voice_prompt:
            torch.save(
                {"cache": self._streaming_state.cache.detach().cpu()},
                splitext(self.voice_prompt)[0] + ".pt",
            )

    def _step_audio_silence(self):
        for _ in range(self.audio_silence_frame_cnt):
            self.step(
                moshi_tokens=self._encode_silence_frame(),
                text_token=self.zero_text_code,
                input_tokens=self._encode_sine_frame(),
            )

    def _step_text_prompt(self):
        if not self.text_prompt_tokens:
            return
        for tok in self.text_prompt_tokens:
            self.step(
                moshi_tokens=self._encode_silence_frame(),
                text_token=tok,
                input_tokens=self._encode_sine_frame(),
            )

    def step_system_prompts(self, mimi):
        """Run full 4-phase prompt injection: voice → silence → text → silence."""
        self._step_voice_prompt(mimi)
        self._step_audio_silence()
        self._step_text_prompt()
        self._step_audio_silence()
