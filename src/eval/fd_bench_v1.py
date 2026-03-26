"""FD-Bench v1.0 evaluation for SteerDuplex.

Evaluates a Moshi checkpoint on Full-Duplex-Bench v1.0 across four tasks:
  - Pause Handling: TOR (lower = better, model should stay silent)
  - Backchannel: TOR (lower = better), Frequency (higher = better)
  - Smooth Turn-Taking: TOR (higher = better), Latency (lower = better)
  - User Interruption: TOR (higher = better), Latency (lower = better),
    optional GPT relevance score (0-5, higher = better)

Pipeline: checkpoint -> streaming inference -> ASR -> per-task scoring

Dataset structure (download from FD-Bench repo Google Drive):
  fd_bench_v1/
    candor_pause_handling/{ID}/input.wav, pause.json, transcription.json
    synthetic_pause_handling/{ID}/input.wav, pause.json, transcription.json
    candor_turn_taking/{ID}/input.wav, turn_taking.json, transcription.json
    icc_backchannel/{ID}/input.wav, transcription.json
    synthetic_user_interruption/{ID}/input.wav, context.wav, interrupt.json

Usage:
    # Standalone eval of a single checkpoint
    python -m eval.fd_bench_v1 \
        --checkpoint runs/full_v3_.../checkpoints/checkpoint_005000/consolidated \
        --data_dir data/benchmarks/fd_bench_v1

    # With sampling for faster eval
    python -m eval.fd_bench_v1 \
        --checkpoint runs/.../consolidated \
        --data_dir data/benchmarks/fd_bench_v1 \
        --max_samples 50
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching FD-Bench v1.0 eval scripts)
# ---------------------------------------------------------------------------
TURN_DURATION_THRESHOLD = 1.0   # seconds — below this, not a real turn
TURN_WORD_THRESHOLD = 3         # at or below this, not a real turn
BACKCHANNEL_MAX_DURATION = 3.0  # seconds — VAD segments longer than this are formal speech

# Task name -> list of data subdirectories that belong to this task
TASK_DIRS = {
    "pause_handling": ["candor_pause_handling", "synthetic_pause_handling"],
    "backchannel": ["icc_backchannel"],
    "turn_taking": ["candor_turn_taking"],
    "user_interruption": ["synthetic_user_interruption"],
}


# ---------------------------------------------------------------------------
# ASR engine
# ---------------------------------------------------------------------------
class ASREngine:
    """Word-level ASR. Tries faster-whisper first, falls back to openai-whisper."""

    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        self.device = device
        self.backend = None
        self.model = None
        self._load(model_size)

    def _load(self, model_size: str):
        # Try faster-whisper (4x faster)
        try:
            from faster_whisper import WhisperModel

            compute = "float16" if "cuda" in self.device else "int8"
            self.model = WhisperModel(
                model_size, device=self.device.split(":")[0], compute_type=compute,
                device_index=int(self.device.split(":")[-1]) if ":" in self.device else 0,
            )
            self.backend = "faster_whisper"
            logger.info("ASR: faster-whisper %s on %s", model_size, self.device)
            return
        except ImportError:
            pass

        # Fallback: openai-whisper
        try:
            import whisper

            self.model = whisper.load_model(model_size, device=self.device)
            self.backend = "whisper"
            logger.info("ASR: openai-whisper %s on %s", model_size, self.device)
            return
        except ImportError:
            pass

        raise RuntimeError(
            "No ASR backend found. Install faster-whisper (pip install faster-whisper) "
            "or openai-whisper (pip install openai-whisper)."
        )

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file.

        Returns:
            {"text": str, "chunks": [{"text": str, "timestamp": [start, end]}, ...]}
        """
        if self.backend == "faster_whisper":
            return self._transcribe_faster(audio_path)
        return self._transcribe_whisper(audio_path)

    def _transcribe_faster(self, audio_path: str) -> dict:
        segments, _ = self.model.transcribe(str(audio_path), word_timestamps=True)
        chunks = []
        words = []
        for seg in segments:
            for w in (seg.words or []):
                text = w.word.strip()
                if text:
                    chunks.append({"text": text, "timestamp": [w.start, w.end]})
                    words.append(text)
        return {"text": " ".join(words), "chunks": chunks}

    def _transcribe_whisper(self, audio_path: str) -> dict:
        result = self.model.transcribe(str(audio_path), word_timestamps=True)
        chunks = []
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                text = w["word"].strip()
                if text:
                    chunks.append({"text": text, "timestamp": [w["start"], w["end"]]})
                    words.append(text)
        return {"text": " ".join(words), "chunks": chunks}


# ---------------------------------------------------------------------------
# VAD (optional, used for backchannel eval)
# ---------------------------------------------------------------------------
class SileroVAD:
    """Silero VAD wrapper for speech segment detection."""

    def __init__(self):
        self.model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True,
        )
        self._get_speech_timestamps = utils[0]

    def get_segments(self, audio_path: str) -> list[dict]:
        """Returns list of {"start": sec, "end": sec, "duration": sec}."""
        audio, sr = sf.read(audio_path)
        audio_t = torch.from_numpy(audio).float()
        if audio_t.ndim > 1:
            audio_t = audio_t[0]

        # Silero VAD expects 16kHz
        if sr != 16000:
            import torchaudio.functional as F
            audio_t = F.resample(audio_t, sr, 16000)
            sr = 16000

        timestamps = self._get_speech_timestamps(audio_t, self.model, sampling_rate=sr)
        segments = []
        for ts in timestamps:
            s = ts["start"] / sr
            e = ts["end"] / sr
            segments.append({"start": s, "end": e, "duration": e - s})
        return segments


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _is_real_turn(chunks: list[dict]) -> bool:
    """Check if ASR output constitutes a real conversational turn.

    FD-Bench threshold: speech duration >= 1s OR word count > 3.
    Below that it's considered a backchannel / noise and not a real turn.
    """
    if not chunks:
        return False

    word_count = len(chunks)
    if word_count > TURN_WORD_THRESHOLD:
        return True

    # Compute speech span from first-word start to last-word end
    duration = chunks[-1]["timestamp"][1] - chunks[0]["timestamp"][0]
    return duration >= TURN_DURATION_THRESHOLD


def _first_word_start(chunks: list[dict]) -> float | None:
    """Get the start time of the first word, or None if no words."""
    if not chunks:
        return None
    return chunks[0]["timestamp"][0]


# ---------------------------------------------------------------------------
# Source split helpers
# ---------------------------------------------------------------------------
def _sample_source(sample: dict) -> str:
    """Extract source label (e.g. 'synthetic', 'candor') from task_dir."""
    task_dir = sample.get("task_dir", "")
    # task_dir looks like 'candor_pause_handling' or 'synthetic_user_interruption'
    parts = task_dir.split("_")
    return parts[0] if parts else "unknown"


# ---------------------------------------------------------------------------
# Per-task scoring
# ---------------------------------------------------------------------------
def score_pause_handling(samples: list[dict]) -> dict:
    """Pause Handling: TOR (Turn-Over Rate). Lower = better.

    The model receives user audio containing pauses. It should NOT start
    speaking during those pauses. TOR=0 if model stays silent or only
    produces a brief backchannel, TOR=1 if model takes a full turn.

    Reports overall TOR plus per-source split (synthetic vs candor).
    """
    tors = []
    tors_by_source: dict[str, list[int]] = {}

    for s in samples:
        chunks = s["asr"]["chunks"]
        tor = 1 if _is_real_turn(chunks) else 0
        tors.append(tor)

        source = _sample_source(s)
        tors_by_source.setdefault(source, []).append(tor)

    metrics = {
        "tor": float(np.mean(tors)) if tors else float("nan"),
        "n_samples": len(tors),
    }
    for source, source_tors in tors_by_source.items():
        metrics[f"{source}_tor"] = float(np.mean(source_tors))
        metrics[f"{source}_n_samples"] = len(source_tors)
    return metrics


def score_backchannel(
    samples: list[dict],
    vad: SileroVAD | None = None,
    gt_distribution_path: str | None = None,
) -> dict:
    """Backchannel: TOR (lower), Frequency (higher), JSD (lower = better).

    The model should produce short acknowledgments (backchannels) but NOT
    take a full turn while the user is speaking.

    JSD measures how closely the model's backchannel timing distribution
    matches the human reference distribution (per-speaker, from
    icc_gt_distribution.json following FD-Bench v1.0 official eval).

    Args:
        gt_distribution_path: Path to icc_gt_distribution.json containing
            per-speaker ground-truth backchannel timing distributions.
    """
    JSD_WINDOW = 0.2  # seconds — bin size (matches FD-Bench official)
    EPSILON = 1e-10

    # Load ground truth distributions if available
    gt_distributions: dict[str, list[float]] = {}
    if gt_distribution_path and Path(gt_distribution_path).exists():
        with open(gt_distribution_path) as f:
            gt_distributions = json.load(f)
        logger.info("Loaded GT backchannel distributions for %d speakers", len(gt_distributions))

    tors = []
    frequencies = []
    jsd_scores = []

    for s in samples:
        chunks = s["asr"]["chunks"]
        output_path = s["output_path"]
        speaker_id = s["id"]  # folder name, used as key into GT distribution

        bc_predictions: list[list[float]] = []  # [start, end] pairs

        if vad is not None:
            segments = vad.get_segments(output_path)
            took_turn = False

            for seg in segments:
                if seg["duration"] > BACKCHANNEL_MAX_DURATION:
                    # Formal speech — model took a full turn
                    took_turn = True
                    break

                # Short segment — check ASR for word content
                seg_chunks = [
                    c for c in chunks
                    if c["timestamp"][0] >= seg["start"] - 0.1
                    and c["timestamp"][1] <= seg["end"] + 0.1
                ]
                word_count = len(seg_chunks)
                if word_count > TURN_WORD_THRESHOLD:
                    took_turn = True
                    break

                if seg["duration"] < TURN_DURATION_THRESHOLD and word_count <= 2:
                    # This is a backchannel
                    bc_predictions.append([seg["start"], seg["end"]])
                else:
                    took_turn = True
                    break

            tors.append(1 if took_turn else 0)

            # Frequency: backchannel count per second
            audio_info = sf.info(output_path)
            total_dur = audio_info.duration
            frequencies.append(len(bc_predictions) / total_dur if total_dur > 0 else 0.0)

            # JSD: per-speaker timing distribution comparison
            gt_dist = gt_distributions.get(speaker_id)
            if gt_dist is not None:
                if not bc_predictions:
                    # No backchannels predicted — maximum divergence
                    jsd_scores.append(1.0)
                else:
                    # Build prediction histogram over 0.2s bins
                    max_time = max(p[1] for p in bc_predictions)
                    n_bins = int(np.ceil(max_time / JSD_WINDOW))
                    if n_bins < 1:
                        n_bins = 1
                    pred_hist = np.zeros(n_bins, dtype=np.float64)
                    for start, end in bc_predictions:
                        bin_start = int(start / JSD_WINDOW)
                        bin_end = int(np.ceil(end / JSD_WINDOW))
                        for b in range(bin_start, min(bin_end, n_bins)):
                            pred_hist[b] += 1

                    # Normalize to probability distribution
                    pred_hist = pred_hist + EPSILON
                    pred_hist = pred_hist / pred_hist.sum()

                    gt_arr = np.array(gt_dist, dtype=np.float64)

                    # Interpolate GT to match prediction length if needed
                    if len(gt_arr) != len(pred_hist):
                        from scipy.interpolate import interp1d
                        x_gt = np.linspace(0, 1, len(gt_arr))
                        x_pred = np.linspace(0, 1, len(pred_hist))
                        interp_fn = interp1d(x_gt, gt_arr, kind="linear", fill_value="extrapolate")
                        gt_arr = interp_fn(x_pred)

                    gt_arr = np.maximum(gt_arr, 0) + EPSILON
                    gt_arr = gt_arr / gt_arr.sum()

                    from scipy.spatial.distance import jensenshannon
                    jsd_val = jensenshannon(pred_hist, gt_arr, base=2)
                    jsd_scores.append(float(jsd_val))
        else:
            # Fallback without VAD
            tors.append(1 if _is_real_turn(chunks) else 0)
            frequencies.append(0.0)

    metrics = {
        "tor": float(np.mean(tors)) if tors else float("nan"),
        "frequency": float(np.mean(frequencies)) if frequencies else float("nan"),
        "n_samples": len(tors),
    }
    if jsd_scores:
        metrics["jsd"] = float(np.mean(jsd_scores))
        metrics["jsd_std"] = float(np.std(jsd_scores))
    else:
        logger.info("JSD: no GT distributions available — skipping JSD metric")

    return metrics


def score_turn_taking(samples: list[dict]) -> dict:
    """Smooth Turn-Taking: TOR (higher = better), Latency (lower = better).

    The model should take the turn promptly after the user finishes speaking.
    Reports overall metrics plus per-source split (e.g. candor).
    """
    tors = []
    latencies = []
    tors_by_source: dict[str, list[int]] = {}
    latencies_by_source: dict[str, list[float]] = {}

    for s in samples:
        chunks = s["asr"]["chunks"]
        # turn_taking.json is a list: [{"text": ..., "timestamp": [start, end]}]
        # Official FD-Bench uses input_turn[0]["timestamp"][0] as the turn boundary
        tt_meta = s.get("turn_taking", [])
        if isinstance(tt_meta, list) and tt_meta and "timestamp" in tt_meta[0]:
            input_end_time = tt_meta[0]["timestamp"][0]
        elif isinstance(tt_meta, dict):
            input_end_time = tt_meta.get("input_end_time", 0)
        else:
            input_end_time = 0
        source = _sample_source(s)

        if _is_real_turn(chunks):
            tors.append(1)
            tors_by_source.setdefault(source, []).append(1)
            start = _first_word_start(chunks)
            if start is not None:
                lat = max(0.0, start - input_end_time)
                latencies.append(lat)
                latencies_by_source.setdefault(source, []).append(lat)
        else:
            tors.append(0)
            tors_by_source.setdefault(source, []).append(0)

    metrics = {
        "tor": float(np.mean(tors)) if tors else float("nan"),
        "latency": float(np.mean(latencies)) if latencies else float("nan"),
        "n_samples": len(tors),
    }
    for source, source_tors in tors_by_source.items():
        metrics[f"{source}_tor"] = float(np.mean(source_tors))
        metrics[f"{source}_n_samples"] = len(source_tors)
        source_lats = latencies_by_source.get(source, [])
        if source_lats:
            metrics[f"{source}_latency"] = float(np.mean(source_lats))
    return metrics


def score_user_interruption(samples: list[dict]) -> dict:
    """User Interruption: TOR (higher = better), Latency (lower = better).

    The user interrupts the assistant. The model should stop, process the
    interruption, and resume with a relevant response.
    """
    tors = []
    latencies = []

    for s in samples:
        chunks = s["asr"]["chunks"]
        interrupt_data = s.get("interrupt", [])
        interrupt_end = (
            interrupt_data[0]["timestamp"][1]
            if interrupt_data and "timestamp" in interrupt_data[0]
            else 0
        )

        if _is_real_turn(chunks):
            tors.append(1)
            start = _first_word_start(chunks)
            if start is not None:
                latencies.append(max(0.0, start - interrupt_end))
        else:
            tors.append(0)

    return {
        "tor": float(np.mean(tors)) if tors else float("nan"),
        "latency": float(np.mean(latencies)) if latencies else float("nan"),
        "n_samples": len(tors),
    }


def score_user_interruption_gpt(
    samples: list[dict], api_key: str | None = None,
) -> dict:
    """GPT-4 relevance scoring for user interruption responses.

    Matches the official FD-Bench v1.0 eval_user_interruption.py exactly:
    - Uses system + user message pair
    - Chain-of-thought analysis before rating
    - Full 0-5 rubric
    - Regex parsing of structured output
    """
    import re

    api_key = api_key or os.environ.get(
        "OPENAI_API_KEY", "sk-WtJy90ltT2hpJSFKQ1u_TA"
    )
    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        "https://litellm-proxy.ml-serving-internal.scale.com/v1",
    )

    try:
        from openai import OpenAI
    except ImportError:
        logger.info("openai package not installed — skipping GPT scoring.")
        return {}

    client = OpenAI(api_key=api_key, base_url=base_url)
    scores = []

    # Official FD-Bench system prompt (verbatim)
    system_prompt = (
        "\n   The scenario is that the user and AI are talking in the spoken conversation.\n"
        "   The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.\n"
        "   Your task is to rate the quality of AI's response after the user interrupt the turn.\n"
        "\n\n"
        "   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):\n"
        "   - 0: The AI's response is totally unrelated to the user's interrupting turn.\n"
        "   - 1: The AI's response is not related to the user's interrupting turn.\n"
        "   - 2: The AI's response is slightly related to the user's interrupting turn.\n"
        "   - 3: The AI's response is related to the user's interrupting turn.\n"
        "   - 4: The AI's response is highly related to the user's interrupting turn.\n"
        "   - 5: The AI's response is perfectly related to the user's interrupting turn.\n"
        "\n\n"
        "   Firstly, briefly analyze the user's interrupting turn and the AI's response\n"
        "   Then, you must return the overall output as the following format:\n"
        "   Analysis: [Your analysis].\n"
        "   I would rate the AI's response as [Rating].\n"
        "   "
    )

    # Regex matching the official eval script
    rating_re = re.compile(
        r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)", re.DOTALL,
    )

    for s in tqdm(samples, desc="  GPT scoring", unit="sample"):
        chunks = s["asr"]["chunks"]
        if not _is_real_turn(chunks):
            continue

        interrupt_data = s.get("interrupt", [])
        if not interrupt_data:
            continue

        # Official field names: "context" and "interrupt" (not "interrupt_text")
        context = interrupt_data[0].get("context", "")
        interrupt_text = interrupt_data[0].get("interrupt", "")
        response_text = s["asr"]["text"]

        # Official FD-Bench user prompt template
        user_prompt = (
            f"\n                - Contextual user turn: {context}\n"
            f"                - User interrupting turn: {interrupt_text}\n"
            f"                - AI's response: {response_text}\n"
            "                "
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                seed=0,
            )
            output = resp.choices[0].message.content.strip()
            match = rating_re.search(output)
            if match:
                score = int(match.group(2))
                scores.append(min(max(score, 0), 5))
            else:
                # Fallback: try to find any digit 0-5
                digits = re.findall(r"\b([0-5])\b", output)
                if digits:
                    scores.append(int(digits[-1]))
                else:
                    logger.warning("Could not parse GPT score for %s: %s",
                                   s.get("id"), output[:100])
        except Exception as e:
            logger.warning("GPT scoring failed for sample %s: %s", s.get("id"), e)

    if scores:
        return {"gpt_score": float(np.mean(scores))}
    return {}


# ---------------------------------------------------------------------------
# Multi-GPU inference worker
# ---------------------------------------------------------------------------
def _inference_worker(
    gpu_id: int,
    sample_indices: list[int],
    sample_dicts: list[dict],
    checkpoint_path: str,
    hf_repo: str,
    system_prompt: str,
    voice_prompt: str | None,
    temperature: float,
    top_k: int,
    step_dir: str,
    result_dict: dict,
):
    """Worker that runs inference for a shard of samples on a single GPU.

    Used by torch.multiprocessing.spawn for multi-GPU parallelism.
    """
    from inference.generate import MoshiInference

    device = f"cuda:{gpu_id}"
    logger.info("GPU %d: loading model on %s (%d samples)", gpu_id, device, len(sample_indices))

    model = MoshiInference(
        hf_repo_id=hf_repo,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    step_dir = Path(step_dir)
    for idx in tqdm(
        sample_indices,
        desc=f"GPU {gpu_id} inference",
        position=gpu_id,
        leave=True,
    ):
        sample = sample_dicts[idx]
        out_dir = step_dir / sample["task_dir"] / sample["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "output.wav"

        if output_path.exists():
            result_dict[idx] = str(output_path)
            continue

        try:
            audio, sr, _text = model.generate(
                user_audio_path=sample["input_path"],
                system_prompt=system_prompt,
                voice_prompt_path=voice_prompt,
                max_duration_sec=600.0,
                temperature=temperature,
                top_k=top_k,
            )

            # Pad/trim to match input length
            input_info = sf.info(sample["input_path"])
            target_samples = int(input_info.duration * sr)
            if len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            elif len(audio) > target_samples:
                audio = audio[:target_samples]

            sf.write(str(output_path), audio, sr)
            result_dict[idx] = str(output_path)
        except Exception as e:
            logger.error("GPU %d: inference failed for %s/%s: %s", gpu_id, sample["task_dir"], sample["id"], e)
            result_dict[idx] = None

    del model
    torch.cuda.empty_cache()
    logger.info("GPU %d: done", gpu_id)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
class FDBenchV1Evaluator:
    """Full-Duplex-Bench v1.0 evaluator.

    Orchestrates: model loading -> batch inference -> ASR -> per-task scoring.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "eval_outputs/fd_bench_v1",
        device: str = "cuda:0",
        system_prompt: str = "You enjoy having a good conversation.",
        voice_prompt: str | None = None,
        max_samples_per_task: int | None = None,
        whisper_model: str = "large-v3",
        hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        temperature: float = 0.8,
        top_k: int = 250,
        inference_devices: list[str] | None = None,
        workers_per_gpu: int = 1,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir)
        self.device = device  # used for ASR/VAD
        self.system_prompt = system_prompt
        self.voice_prompt = voice_prompt
        self.max_samples = max_samples_per_task
        self.whisper_model = whisper_model
        self.hf_repo = hf_repo
        self.temperature = temperature
        self.top_k = top_k
        # Multi-GPU: list of devices for inference (e.g. ["cuda:0", "cuda:1"])
        # Falls back to single-device [self.device] if not specified.
        self.inference_devices = inference_devices or [device]
        self.workers_per_gpu = workers_per_gpu

        self._model = None
        self._asr = None
        self._vad = None

    # -- Lazy-loaded components --

    def _ensure_asr(self):
        if self._asr is None:
            self._asr = ASREngine(model_size=self.whisper_model, device=self.device)

    def _ensure_vad(self):
        if self._vad is None:
            try:
                self._vad = SileroVAD()
            except Exception as e:
                logger.warning("Silero VAD unavailable: %s (backchannel eval degraded)", e)

    def _run_inference_multi_gpu(
        self, samples: list[dict], checkpoint_path: str, step_dir: Path,
    ):
        """Run inference across multiple GPUs with multiple workers per GPU."""
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        # Parse GPU IDs from inference_devices
        gpu_ids = []
        for dev in self.inference_devices:
            if ":" in dev:
                gpu_ids.append(int(dev.split(":")[-1]))
            else:
                gpu_ids.append(0)

        # Build worker list: (worker_id, gpu_id) for each worker
        workers = []
        for gpu_id in gpu_ids:
            for _ in range(self.workers_per_gpu):
                workers.append(gpu_id)

        n_workers = len(workers)

        # Shard samples across all workers (round-robin)
        shards: list[list[int]] = [[] for _ in range(n_workers)]
        for i in range(len(samples)):
            shards[i % n_workers].append(i)

        logger.info(
            "Parallel inference: %d GPUs x %d workers/GPU = %d workers, "
            "%d total samples (%s samples/worker)",
            len(gpu_ids), self.workers_per_gpu, n_workers,
            len(samples), [len(s) for s in shards],
        )

        # Shared dict for results
        manager = mp.Manager()
        result_dict = manager.dict()

        # Serialize sample dicts (can't pass complex objects to workers)
        sample_dicts = [
            {k: v for k, v in s.items() if k != "asr"} for s in samples
        ]

        processes = []
        for worker_idx, (gpu_id, shard) in enumerate(zip(workers, shards)):
            if not shard:
                continue
            p = mp.Process(
                target=_inference_worker,
                args=(
                    gpu_id, shard, sample_dicts, checkpoint_path,
                    self.hf_repo, self.system_prompt, self.voice_prompt,
                    self.temperature, self.top_k, str(step_dir), result_dict,
                ),
                name=f"worker-{worker_idx}-gpu{gpu_id}",
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Collect results back into samples
        for idx, output_path in result_dict.items():
            samples[idx]["output_path"] = output_path

        logger.info("Parallel inference complete: %d/%d succeeded",
                     sum(1 for s in samples if s.get("output_path")), len(samples))

    def _load_model(self, checkpoint_path: str):
        """Load Moshi model from a checkpoint directory."""
        from inference.generate import MoshiInference

        # Free previous model
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()

        logger.info("Loading model from %s on %s", checkpoint_path, self.device)
        self._model = MoshiInference(
            hf_repo_id=self.hf_repo,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

    # -- Sample discovery --

    def _discover_samples(self, task_dirs: list[str]) -> list[dict]:
        """Find all samples for a task from the data directory."""
        samples = []
        for dir_name in task_dirs:
            task_path = self.data_dir / dir_name
            if not task_path.exists():
                logger.warning("Task dir not found: %s", task_path)
                continue

            for sample_dir in sorted(task_path.iterdir()):
                if not sample_dir.is_dir():
                    continue
                input_wav = sample_dir / "input.wav"
                if not input_wav.exists():
                    continue

                sample = {
                    "id": sample_dir.name,
                    "task_dir": dir_name,
                    "input_path": str(input_wav),
                    "sample_dir": str(sample_dir),
                }

                # Load all available metadata files
                for meta_name in [
                    "pause.json", "turn_taking.json", "interrupt.json",
                    "transcription.json",
                ]:
                    meta_path = sample_dir / meta_name
                    if meta_path.exists():
                        with open(meta_path) as f:
                            key = meta_name.replace(".json", "")
                            sample[key] = json.load(f)

                samples.append(sample)

        # Deterministic subsampling
        if self.max_samples and len(samples) > self.max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(samples), self.max_samples, replace=False)
            samples = [samples[i] for i in sorted(indices)]

        return samples

    # -- Inference --

    def _run_inference_sample(self, sample: dict, step_dir: Path) -> str:
        """Generate time-synchronous output for one sample. Returns output path."""
        out_dir = step_dir / sample["task_dir"] / sample["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "output.wav"

        if output_path.exists():
            logger.debug("Skipping (cached): %s", output_path)
            return str(output_path)

        audio, sr = self._model.generate(
            user_audio_path=sample["input_path"],
            system_prompt=self.system_prompt,
            voice_prompt_path=self.voice_prompt,
            max_duration_sec=600.0,
            temperature=self.temperature,
            top_k=self.top_k,
        )

        # Pad/trim to match input length (critical for time-synchronous eval)
        input_info = sf.info(sample["input_path"])
        target_samples = int(input_info.duration * sr)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        elif len(audio) > target_samples:
            audio = audio[:target_samples]

        sf.write(str(output_path), audio, sr)
        return str(output_path)

    # -- ASR --

    def _run_asr_sample(self, sample: dict, task_name: str) -> dict:
        """Run ASR on one sample's output. Handles interrupt cropping."""
        output_path = sample["output_path"]

        # For user interruption: crop output after the interruption ends,
        # then ASR only the response portion
        if task_name == "user_interruption":
            interrupt_data = sample.get("interrupt", [])
            if interrupt_data and "timestamp" in interrupt_data[0]:
                interrupt_end = interrupt_data[0]["timestamp"][1]
                audio, sr = sf.read(output_path)
                crop_sample = int(interrupt_end * sr)
                if 0 < crop_sample < len(audio):
                    cropped = audio[crop_sample:]
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False,
                    ) as tmp:
                        sf.write(tmp.name, cropped, sr)
                        result = self._asr.transcribe(tmp.name)
                        os.unlink(tmp.name)
                    # Offset timestamps to absolute time
                    for chunk in result["chunks"]:
                        chunk["timestamp"][0] += interrupt_end
                        chunk["timestamp"][1] += interrupt_end
                    return result

        return self._asr.transcribe(output_path)

    # -- Main evaluate entry point --

    def evaluate(self, checkpoint_path: str, step: int | None = None) -> dict:
        """Run full FD-Bench v1.0 evaluation.

        Args:
            checkpoint_path: Path to checkpoint directory containing
                consolidated.safetensors (e.g., .../checkpoint_005000/consolidated).
            step: Training step number (for output organization).

        Returns:
            Flat dict of metrics suitable for wandb.log(), e.g.:
            {
                "fd_bench/pause_handling/tor": 0.12,
                "fd_bench/turn_taking/tor": 0.87,
                "fd_bench/turn_taking/latency": 0.43,
                ...
            }
        """
        step_name = f"step_{step}" if step is not None else "latest"
        step_dir = self.output_dir / step_name

        # Resume: if this step was fully evaluated before, return cached metrics
        cached_metrics_path = step_dir / "metrics.json"
        if cached_metrics_path.exists():
            try:
                with open(cached_metrics_path) as f:
                    cached = json.load(f)
                # Remove non-metric keys if present
                cached.pop("step", None)
                logger.info("Loaded cached metrics for %s (%d metrics)", step_name, len(cached))
                return cached
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupted cached metrics for %s — re-evaluating", step_name)

        self._ensure_asr()
        self._ensure_vad()

        # Collect all samples across all tasks first (for multi-GPU batching)
        all_metrics: dict[str, float] = {}
        task_samples: dict[str, list[dict]] = {}

        for task_name, task_dirs in TASK_DIRS.items():
            samples = self._discover_samples(task_dirs)
            if not samples:
                logger.warning("No samples found for %s — skipping", task_name)
                continue
            task_samples[task_name] = samples

        # Phase 1: Inference (multi-GPU)
        all_samples = []
        sample_task_map = []  # track which task each sample belongs to
        for task_name, samples in task_samples.items():
            for s in samples:
                all_samples.append(s)
                sample_task_map.append(task_name)

        # Pre-populate output_path for any already-cached inference results
        cached_count = 0
        for s in all_samples:
            cached_path = step_dir / s["task_dir"] / s["id"] / "output.wav"
            if cached_path.exists():
                s["output_path"] = str(cached_path)
                cached_count += 1

        need_inference = [s for s in all_samples if not s.get("output_path")]
        logger.info("Phase 1: Inference — %d total, %d cached, %d to run",
                     len(all_samples), cached_count, len(need_inference))

        if need_inference:
            total_workers = len(self.inference_devices) * self.workers_per_gpu
            if total_workers > 1:
                self._run_inference_multi_gpu(
                    need_inference, checkpoint_path, step_dir,
                )
            else:
                self._load_model(checkpoint_path)
                for sample in tqdm(need_inference, desc="Inference", unit="sample"):
                    try:
                        sample["output_path"] = self._run_inference_sample(sample, step_dir)
                    except Exception as e:
                        logger.error("  Inference failed for %s/%s: %s",
                                     sample["task_dir"], sample["id"], e)
                        sample["output_path"] = None
                # Free single-GPU model
                del self._model
                self._model = None
                torch.cuda.empty_cache()
        else:
            logger.info("  All inference cached — skipping")

        # Phase 2 & 3: ASR + Scoring per task
        for task_name, samples in task_samples.items():
            logger.info("=== Task: %s ===", task_name)

            # Filter out failed inference
            samples = [s for s in samples if s.get("output_path")]
            logger.info("  %d samples with inference output", len(samples))

            # Phase 2: ASR (with caching — skip if output.json already exists)
            logger.info("  Phase 2: ASR...")
            asr_cached = 0
            for sample in tqdm(samples, desc=f"  ASR ({task_name})", unit="sample"):
                asr_path = Path(sample["output_path"]).parent / "output.json"

                # Resume: load cached ASR if available
                if asr_path.exists():
                    try:
                        with open(asr_path) as f:
                            sample["asr"] = json.load(f)
                        asr_cached += 1
                        continue
                    except (json.JSONDecodeError, KeyError):
                        pass  # corrupted cache — re-run ASR

                try:
                    sample["asr"] = self._run_asr_sample(sample, task_name)
                except Exception as e:
                    logger.error("    ASR failed for %s/%s: %s",
                                 sample["task_dir"], sample["id"], e)
                    sample["asr"] = {"text": "", "chunks": []}

                with open(asr_path, "w") as f:
                    json.dump(sample["asr"], f, indent=2)

            if asr_cached:
                logger.info("  ASR: %d cached, %d new", asr_cached, len(samples) - asr_cached)

            task_samples[task_name] = samples  # update with filtered list

            # Phase 3: Scoring
            logger.info("  Phase 3: Scoring...")
            if task_name == "pause_handling":
                metrics = score_pause_handling(samples)
            elif task_name == "backchannel":
                gt_dist_path = self.data_dir / "icc_gt_distribution.json"
                metrics = score_backchannel(
                    samples, vad=self._vad,
                    gt_distribution_path=str(gt_dist_path),
                )
            elif task_name == "turn_taking":
                metrics = score_turn_taking(samples)
            elif task_name == "user_interruption":
                metrics = score_user_interruption(samples)
                # Optional GPT scoring
                gpt_metrics = score_user_interruption_gpt(samples)
                metrics.update(gpt_metrics)
            else:
                metrics = {}

            # Flatten with prefix
            for k, v in metrics.items():
                all_metrics[f"fd_bench/{task_name}/{k}"] = v

            logger.info("  Results: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
        torch.cuda.empty_cache()

        # Save metrics for resumability
        step_dir.mkdir(parents=True, exist_ok=True)
        with open(step_dir / "metrics.json", "w") as f:
            json.dump({"step": step, **all_metrics}, f, indent=2)

        logger.info("=== FD-Bench v1.0 evaluation complete ===")
        for k, v in sorted(all_metrics.items()):
            if isinstance(v, float) and not k.endswith("n_samples"):
                logger.info("  %s: %.4f", k, v)

        return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FD-Bench v1.0 evaluation for SteerDuplex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint dir (containing consolidated.safetensors)",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to FD-Bench v1.0 dataset root",
    )
    parser.add_argument("--output_dir", default="eval_outputs/fd_bench_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per task (default: all)")
    parser.add_argument("--system_prompt", default="You enjoy having a good conversation.")
    parser.add_argument("--voice_prompt", default=None)
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--hf_repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--step", type=int, default=None,
                        help="Training step (for output dir naming)")
    parser.add_argument("--inference_devices", nargs="+", default=None,
                        help="GPU devices for parallel inference (e.g. cuda:0 cuda:1 cuda:2)")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Number of parallel workers per GPU (default: 1, use 3 for 80GB GPUs)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    evaluator = FDBenchV1Evaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        system_prompt=args.system_prompt,
        voice_prompt=args.voice_prompt,
        max_samples_per_task=args.max_samples,
        whisper_model=args.whisper_model,
        hf_repo=args.hf_repo,
        temperature=args.temperature,
        top_k=args.top_k,
        inference_devices=args.inference_devices,
        workers_per_gpu=args.workers_per_gpu,
    )

    metrics = evaluator.evaluate(args.checkpoint, step=args.step)

    # Save metrics to JSON
    out_path = Path(args.output_dir) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
