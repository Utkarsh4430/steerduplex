"""Whisper transcription helpers for evaluation outputs."""

import logging
from typing import Dict, List, Union

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from transformers import pipeline as hf_pipeline
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _fallback_audio() -> Dict[str, Union[np.ndarray, int]]:
    return {
        "array": np.zeros(16000, dtype=np.float32),
        "sampling_rate": 16000,
    }


def transcribe_batch(
    audio_paths: List[str],
    model_name: str = "openai/whisper-large-v3",
    batch_size: int = 8,
    device: str = "cuda",
) -> List[str]:
    """Transcribe WAV files with Whisper and preserve input ordering."""
    if not audio_paths:
        return []

    logger.info("Loading Whisper model %s on %s...", model_name, device)
    torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype,
        chunk_length_s=30,
        stride_length_s=5,
    )

    transcriptions = []
    for start_idx in tqdm(
        range(0, len(audio_paths), batch_size),
        desc="Whisper transcription",
        unit="batch",
    ):
        batch_paths = audio_paths[start_idx : start_idx + batch_size]
        batch_audio = []
        for path in batch_paths:
            try:
                audio, sample_rate = sf.read(path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                target_sr = 16000
                if sample_rate != target_sr:
                    g = gcd(sample_rate, target_sr)
                    audio = resample_poly(audio, target_sr // g, sample_rate // g).astype(np.float32)
                    sample_rate = target_sr
                batch_audio.append(
                    {
                        "array": np.asarray(audio, dtype=np.float32),
                        "sampling_rate": sample_rate,
                    }
                )
            except Exception as exc:
                logger.warning("Failed to load %s for Whisper: %s", path, exc)
                batch_audio.append(_fallback_audio())

        try:
            results = asr(batch_audio, batch_size=len(batch_audio))
        except Exception as exc:
            logger.exception("Whisper batch failed for items %d-%d", start_idx, start_idx + len(batch_audio) - 1)
            transcriptions.extend([""] * len(batch_audio))
            logger.warning("Returning empty transcriptions for failed batch: %s", exc)
            continue

        if isinstance(results, dict):
            results = [results]

        for result in results:
            text = result["text"] if isinstance(result, dict) else str(result)
            transcriptions.append(text.strip())

    if len(transcriptions) != len(audio_paths):
        raise RuntimeError(
            f"Whisper returned {len(transcriptions)} transcripts for {len(audio_paths)} files"
        )

    return transcriptions
