"""Audio MultiChallenge (AudioMC) evaluation for SteerDuplex.

452 multi-turn conversations testing 4 axes:
  Inference Memory, Instruction Retention, Self Coherence, Voice Editing

Pipeline:
  1. Pre-generate user audio from transcripts via Qwen3-TTS (cached)
  2. Multi-GPU inference: feed user audio → Moshi → agent audio
  3. ASR on agent output
  4. LLM rubric judge (parallel with retries)

Dataset: HuggingFace ScaleAI/audiomc

Usage:
    python -m eval.audio_multichallenge \
        --checkpoint runs/.../checkpoint_005000/consolidated \
        --device cuda:0 --max_samples 50

    # Multi-GPU
    python -m eval.audio_multichallenge \
        --checkpoint runs/.../checkpoint_005000/consolidated \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 --max_samples 0
"""

import argparse
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

RUBRIC_JUDGE_SYSTEM = (
    "You are evaluating an AI assistant's response against specific rubrics. "
    "For each rubric, determine if the assistant's response PASSES or FAILS. "
    "Return a JSON object mapping each rubric to true (pass) or false (fail). "
    "Only return the JSON, nothing else."
)

RUBRIC_JUDGE_USER = (
    "Conversation context:\n{context}\n\n"
    "Assistant's latest response:\n{response}\n\n"
    "Rubrics to evaluate:\n{rubrics}\n\n"
    "Return JSON like {{\"rubric_1\": true, \"rubric_2\": false}}."
)


# ---------------------------------------------------------------------------
# Step 1: TTS — generate user audio from transcripts (cached)
# ---------------------------------------------------------------------------
def tts_generate_user_audio(
    samples: list[dict],
    cache_dir: Path,
    tts_device: str = "cuda:0",
    num_workers: int = 1,
):
    """Generate user audio for all samples using Qwen3-TTS. Cached to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check what needs generating
    to_generate = []
    for i, sample in enumerate(samples):
        wav_path = cache_dir / f"sample_{i}.wav"
        if wav_path.exists():
            sample["tts_audio_path"] = str(wav_path)
        else:
            to_generate.append((i, sample))

    if not to_generate:
        logger.info("All %d TTS samples cached", len(samples))
        return

    logger.info("Generating TTS for %d/%d samples on %s...", len(to_generate), len(samples), tts_device)

    from qwen_tts import Qwen3TTSModel

    # Use CustomVoice model with preset speaker (no ref audio needed)
    tts = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=tts_device, dtype=torch.bfloat16,
    )

    for idx, sample in tqdm(to_generate, desc="TTS"):
        user_texts = []
        for turn_i in range(1, 9):
            key = f"user_turn_{turn_i}_transcript"
            if key in sample and sample[key]:
                user_texts.append(sample[key])

        if not user_texts:
            continue

        combined_text = " ... ".join(user_texts)
        wav_path = cache_dir / f"sample_{idx}.wav"

        try:
            wavs, sr = tts.generate_custom_voice(
                text=combined_text, language="English", speaker="Ryan",
            )
            audio = wavs[0] if isinstance(wavs, list) else wavs
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            if audio.ndim > 1:
                audio = audio[0]
            sf.write(str(wav_path), audio.astype(np.float32), sr)
            sample["tts_audio_path"] = str(wav_path)
        except Exception as e:
            logger.warning("TTS failed for sample %d: %s", idx, e)

    del tts
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 2: Load dataset
# ---------------------------------------------------------------------------
def load_dataset_audiomc():
    """Load AudioMC from HuggingFace (metadata only, no audio decoding)."""
    from datasets import load_dataset

    logger.info("Loading AudioMC from HuggingFace...")
    ds = load_dataset("ScaleAI/audiomc", split="test")

    # Remove audio columns to avoid torchcodec issues
    audio_cols = [c for c in ds.column_names if "audio" in c.lower()]
    if audio_cols:
        ds = ds.remove_columns(audio_cols)

    samples = [dict(row) for row in ds]
    logger.info("Loaded %d conversations", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Step 3: LLM judge with retries
# ---------------------------------------------------------------------------
def _make_llm_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "sk-WtJy90ltT2hpJSFKQ1u_TA"),
        base_url=os.environ.get("OPENAI_BASE_URL",
                                "https://litellm-proxy.ml-serving-internal.scale.com/v1"),
    )


def judge_rubrics_with_retry(
    context: str, response: str, rubrics: str,
    model: str = "openai/gpt-5.1-chat-latest",
    max_retries: int = 5,
) -> dict:
    """LLM judge with retry on rate limit."""
    import re
    client = _make_llm_client()

    for attempt in range(max_retries):
        try:
            # GPT-5.x only supports temperature=1
            temp = 1.0 if "gpt-5" in model else 0

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RUBRIC_JUDGE_SYSTEM},
                    {"role": "user", "content": RUBRIC_JUDGE_USER.format(
                        context=context, response=response, rubrics=rubrics,
                    )},
                ],
                max_tokens=500,
                temperature=temp,
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {}
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 5 * (attempt + 1)
                logger.debug("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
            else:
                logger.warning("Judge failed (attempt %d): %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(2)
    return {}


# ---------------------------------------------------------------------------
# Step 4: Inference worker (per GPU)
# ---------------------------------------------------------------------------
def _inference_worker(
    gpu_id: int,
    sample_indices: list[int],
    samples: list[dict],
    checkpoint_path: str,
    hf_repo: str,
    whisper_model: str,
) -> dict:
    """Run inference on a shard of samples on one GPU. Returns {idx: response_text}."""
    from inference.generate import MoshiInference

    device = f"cuda:{gpu_id}"
    logger.info("GPU %d: loading model for %d samples", gpu_id, len(sample_indices))

    model = MoshiInference(
        hf_repo_id=hf_repo,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # ASR
    try:
        from faster_whisper import WhisperModel
        asr = WhisperModel(whisper_model, device="cuda", compute_type="float16",
                           device_index=gpu_id)
    except ImportError:
        import whisper
        asr = whisper.load_model(whisper_model, device=device)

    results = {}
    for idx in tqdm(sample_indices, desc=f"GPU {gpu_id}", position=gpu_id):
        sample = samples[idx]
        audio_path = sample.get("tts_audio_path")
        if not audio_path or not Path(audio_path).exists():
            results[idx] = ""
            continue

        try:
            out_audio, out_sr, text_tokens = model.generate(
                user_audio_path=audio_path,
                system_prompt="You are a helpful voice assistant. Follow instructions carefully.",
                max_duration_sec=60.0,
            )

            # ASR
            if len(out_audio) > 0 and np.abs(out_audio).max() > 0.001:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, out_audio, out_sr)
                    tmp = f.name
                try:
                    if hasattr(asr, 'model_size_or_path'):
                        segs, _ = asr.transcribe(tmp)
                        response = " ".join(s.text.strip() for s in segs)
                    else:
                        r = asr.transcribe(tmp, language="en")
                        response = r["text"].strip()
                finally:
                    os.unlink(tmp)
            else:
                response = " ".join(t for t in text_tokens if not t.startswith("["))

            results[idx] = response
        except Exception as e:
            logger.warning("GPU %d: failed sample %d: %s", gpu_id, idx, e)
            results[idx] = ""

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Step 5: Main evaluation
# ---------------------------------------------------------------------------
def evaluate_checkpoint(
    checkpoint_path: str,
    samples: list[dict],
    devices: list[str],
    max_samples: int | None = None,
    hf_repo: str = "kyutai/moshiko-pytorch-bf16",
    whisper_model: str = "large-v3",
    judge_workers: int = 16,
) -> dict:
    """Evaluate a single checkpoint on AudioMC."""

    eval_samples = list(range(len(samples)))
    if max_samples and max_samples > 0 and len(eval_samples) > max_samples:
        import random
        random.seed(42)
        eval_samples = random.sample(eval_samples, max_samples)

    # Filter to samples with TTS audio
    eval_samples = [i for i in eval_samples if samples[i].get("tts_audio_path")]
    logger.info("Evaluating %d samples with TTS audio", len(eval_samples))

    if not eval_samples:
        return {"audiomc/apr": 0, "audiomc/ars": 0, "audiomc/n_samples": 0}

    # Multi-GPU inference
    n_gpus = len(devices)
    gpu_ids = [int(d.split(":")[-1]) if ":" in d else 0 for d in devices]

    if n_gpus > 1:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        shards = [[] for _ in range(n_gpus)]
        for i, idx in enumerate(eval_samples):
            shards[i % n_gpus].append(idx)

        manager = mp.Manager()
        all_results = manager.dict()

        processes = []
        for worker_idx, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
            if not shard:
                continue
            p = mp.Process(
                target=lambda gid, s, r: r.update(
                    _inference_worker(gid, s, samples, checkpoint_path, hf_repo, whisper_model)
                ),
                args=(gpu_id, shard, all_results),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        responses = dict(all_results)
    else:
        gpu_id = gpu_ids[0]
        responses = _inference_worker(
            gpu_id, eval_samples, samples, checkpoint_path, hf_repo, whisper_model,
        )

    # LLM judge (parallel with retries)
    logger.info("Running LLM judge on %d responses (%d workers)...", len(responses), judge_workers)

    results_by_axis = {}
    all_rubric_pass = 0
    all_rubric_total = 0
    all_conv_pass = 0

    def _judge_one(idx):
        sample = samples[idx]
        response = responses.get(idx, "")
        if not response:
            return sample.get("axis", "unknown"), {}, False

        # Build context
        context_parts = []
        for turn_i in range(1, 9):
            ut = sample.get(f"user_turn_{turn_i}_transcript", "")
            at = sample.get(f"assistant_turn_{turn_i}_transcript", "")
            if ut:
                context_parts.append(f"User: {ut}")
            if at:
                context_parts.append(f"Assistant: {at}")
        context = "\n".join(context_parts)
        rubrics = sample.get("rubric", "")
        axis = sample.get("axis", "unknown")

        rubric_results = judge_rubrics_with_retry(context, response, rubrics)
        all_pass = all(v is True for v in rubric_results.values()) if rubric_results else False
        return axis, rubric_results, all_pass

    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        futures = {pool.submit(_judge_one, idx): idx for idx in responses}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            try:
                axis, rubric_results, all_pass = future.result()

                if axis not in results_by_axis:
                    results_by_axis[axis] = {"rp": 0, "rt": 0, "cp": 0, "ct": 0}
                results_by_axis[axis]["ct"] += 1

                n_pass = sum(1 for v in rubric_results.values() if v is True)
                n_total = max(len(rubric_results), 1)

                results_by_axis[axis]["rp"] += n_pass
                results_by_axis[axis]["rt"] += n_total
                all_rubric_pass += n_pass
                all_rubric_total += n_total

                if all_pass:
                    results_by_axis[axis]["cp"] += 1
                    all_conv_pass += 1
            except Exception as e:
                logger.warning("Judge error: %s", e)

    total = len(eval_samples)
    metrics = {
        "audiomc/apr": all_conv_pass / max(total, 1),
        "audiomc/ars": all_rubric_pass / max(all_rubric_total, 1),
        "audiomc/n_samples": total,
    }
    for axis, r in results_by_axis.items():
        metrics[f"audiomc/{axis}/apr"] = r["cp"] / max(r["ct"], 1)
        metrics[f"audiomc/{axis}/ars"] = r["rp"] / max(r["rt"], 1)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AudioMC evaluation")
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"],
                        help="GPU devices for inference (e.g. cuda:0 cuda:1)")
    parser.add_argument("--tts_device", default=None,
                        help="GPU for TTS generation (default: first inference device)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples (0 or None = all)")
    parser.add_argument("--hf_repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--judge_workers", type=int, default=16)
    parser.add_argument("--output", default="eval_outputs/audio_multichallenge")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load dataset
    samples = load_dataset_audiomc()

    # Pre-generate TTS audio (cached)
    cache_dir = Path(args.output) / "tts_cache"
    tts_device = args.tts_device or args.devices[0]
    tts_generate_user_audio(samples, cache_dir, tts_device=tts_device)

    # Evaluate each checkpoint
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in args.checkpoint:
        step = "unknown"
        for part in Path(ckpt_path).parts:
            if "checkpoint_" in part or "step_" in part:
                step = part.split("_")[-1]
                break

        logger.info("=" * 60)
        logger.info("Evaluating: %s (step %s)", ckpt_path, step)
        logger.info("=" * 60)

        metrics = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            samples=samples,
            devices=args.devices,
            max_samples=args.max_samples if args.max_samples and args.max_samples > 0 else None,
            hf_repo=args.hf_repo,
            whisper_model=args.whisper_model,
            judge_workers=args.judge_workers,
        )

        metrics_path = output_dir / f"step_{step}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"step": step, **metrics}, f, indent=2)

        print(f"\nStep {step} — AudioMC Results")
        print(f"{'='*50}")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
