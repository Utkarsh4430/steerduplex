"""Big Bench Audio evaluation for SteerDuplex.

1000 audio reasoning questions across 4 categories (250 each):
  - Formal Fallacies, Navigate, Object Counting, Web of Lies

Pipeline: checkpoint → inference (audio→audio) → ASR → LLM judge scoring
Dataset: HuggingFace ArtificialAnalysis/big_bench_audio

Usage:
    python -m eval.big_bench_audio \
        --checkpoint runs/full_v3_.../checkpoints/checkpoint_005000/consolidated \
        --device cuda:0 --max_samples 50

    # Evaluate multiple checkpoints
    python -m eval.big_bench_audio \
        --checkpoint runs/.../checkpoint_001000/consolidated \
                     runs/.../checkpoint_005000/consolidated \
                     runs/.../checkpoint_009000/consolidated
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

# LLM judge prompt (following Big Bench Audio protocol)
JUDGE_SYSTEM = (
    "You are an expert evaluator. Assess whether the CANDIDATE ANSWER "
    "is CORRECT or INCORRECT compared to the OFFICIAL ANSWER. "
    "The candidate may paraphrase or restate the answer differently — "
    "that's fine as long as the meaning matches. "
    "Respond with exactly one word: CORRECT or INCORRECT."
)

JUDGE_USER = (
    "OFFICIAL ANSWER: {official_answer}\n\n"
    "CANDIDATE ANSWER: {candidate_answer}\n\n"
    "Is the candidate answer CORRECT or INCORRECT?"
)


def load_dataset_local(data_dir: str = "data/benchmarks/big_bench_audio"):
    """Load Big Bench Audio from local files.

    Expects:
        data_dir/metadata.jsonl — one JSON per line with category, official_answer, file_name, id
        data_dir/data/question_*.mp3 — audio files
    """
    import json
    data_path = Path(data_dir)
    metadata_path = data_path / "metadata.jsonl"

    if not metadata_path.exists():
        # Auto-download from HuggingFace
        logger.info("Downloading Big Bench Audio metadata + audio...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            "ArtificialAnalysis/big_bench_audio", repo_type="dataset",
            local_dir=str(data_path),
        )

    with open(metadata_path) as f:
        entries = [json.loads(l) for l in f]

    # Store audio file paths (don't load into memory — MP3 needs special handling)
    samples = []
    for entry in entries:
        audio_path = data_path / entry["file_name"]
        if audio_path.exists():
            entry["audio_path"] = str(audio_path)
            samples.append(entry)

    logger.info("Loaded %d/%d Big Bench Audio samples from %s", len(samples), len(entries), data_dir)
    return samples


def run_inference(model, audio_array, sr, system_prompt="Answer the question concisely."):
    """Run Moshi inference on an audio sample. Returns (output_audio, sr, text_tokens)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Resample to model's sample rate if needed
        if sr != model.sample_rate:
            import torchaudio.functional as F
            audio_t = torch.from_numpy(audio_array).float()
            if audio_t.ndim == 1:
                audio_t = audio_t.unsqueeze(0)
            audio_t = F.resample(audio_t, sr, model.sample_rate)
            audio_array = audio_t.squeeze().numpy()
            sr = model.sample_rate
        sf.write(f.name, audio_array, sr)
        tmp_path = f.name

    try:
        audio_out, out_sr, text_tokens = model.generate(
            user_audio_path=tmp_path,
            system_prompt=system_prompt,
            max_duration_sec=30.0,
        )
        return audio_out, out_sr, text_tokens
    finally:
        os.unlink(tmp_path)


def transcribe(audio, sr, asr_model):
    """Transcribe audio using faster-whisper or openai-whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        tmp_path = f.name

    try:
        if hasattr(asr_model, 'transcribe') and hasattr(asr_model, 'model_size_or_path'):
            # faster-whisper
            segments, _ = asr_model.transcribe(tmp_path)
            return " ".join(seg.text.strip() for seg in segments)
        else:
            # openai-whisper
            result = asr_model.transcribe(tmp_path, language="en")
            return result["text"].strip()
    finally:
        os.unlink(tmp_path)


def judge_answer(official: str, candidate: str, client, model: str = "openai/gpt-5.1-chat-latest") -> bool:
    """Use LLM judge to score correctness."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER.format(
                    official_answer=official,
                    candidate_answer=candidate,
                )},
            ],
            max_tokens=10,
            temperature=0,
        )
        result = resp.choices[0].message.content.strip().upper()
        return "CORRECT" in result
    except Exception as e:
        logger.warning("Judge failed: %s", e)
        return False


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset,
    device: str = "cuda:0",
    max_samples: int | None = None,
    hf_repo: str = "kyutai/moshiko-pytorch-bf16",
    whisper_model: str = "large-v3",
) -> dict:
    """Evaluate a single checkpoint on Big Bench Audio."""
    from inference.generate import MoshiInference

    logger.info("Loading model from %s on %s", checkpoint_path, device)
    model = MoshiInference(
        hf_repo_id=hf_repo,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Load ASR
    logger.info("Loading ASR (faster-whisper %s)...", whisper_model)
    try:
        from faster_whisper import WhisperModel
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        asr = WhisperModel(whisper_model, device="cuda", compute_type="float16",
                           device_index=device_idx)
    except ImportError:
        import whisper
        asr = whisper.load_model(whisper_model, device=device)

    # LLM judge
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "sk-WtJy90ltT2hpJSFKQ1u_TA"),
        base_url=os.environ.get("OPENAI_BASE_URL",
                                "https://litellm-proxy.ml-serving-internal.scale.com/v1"),
    )

    # Run evaluation
    samples = list(dataset)
    if max_samples and len(samples) > max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)

    results_by_category = {}
    all_correct = 0
    all_total = 0

    for sample in tqdm(samples, desc="Big Bench Audio"):
        category = sample["category"]
        official_answer = sample["official_answer"]
        audio_path = sample["audio_path"]

        try:
            # Inference directly from file path
            out_audio, out_sr, text_tokens = model.generate(
                user_audio_path=audio_path,
                system_prompt="Listen carefully and answer the question. Be concise.",
                max_duration_sec=30.0,
            )

            # Transcribe model output
            if len(out_audio) > 0 and np.abs(out_audio).max() > 0.001:
                candidate = transcribe(out_audio, out_sr, asr)
            else:
                # Use text tokens if no audio
                candidate = " ".join(t for t in text_tokens if not t.startswith("["))

            # Judge
            correct = judge_answer(official_answer, candidate, client)

        except Exception as e:
            logger.warning("Failed on %s: %s", sample.get("id", "?"), e)
            candidate = ""
            correct = False

        if category not in results_by_category:
            results_by_category[category] = {"correct": 0, "total": 0}
        results_by_category[category]["total"] += 1
        if correct:
            results_by_category[category]["correct"] += 1
            all_correct += 1
        all_total += 1

    # Compute metrics
    metrics = {
        "bba/overall_accuracy": all_correct / max(all_total, 1),
        "bba/n_samples": all_total,
    }
    for cat, res in results_by_category.items():
        acc = res["correct"] / max(res["total"], 1)
        metrics[f"bba/{cat}/accuracy"] = acc
        metrics[f"bba/{cat}/n_samples"] = res["total"]

    # Clean up
    del model
    torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Big Bench Audio evaluation")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="Checkpoint path(s)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--hf_repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--data_dir", default="data/benchmarks/big_bench_audio",
                        help="Path to Big Bench Audio dataset")
    parser.add_argument("--output", default="eval_outputs/big_bench_audio")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset = load_dataset_local(args.data_dir)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in args.checkpoint:
        # Extract step from path
        step = "unknown"
        for part in Path(ckpt_path).parts:
            if "checkpoint_" in part or "step_" in part:
                step = part.split("_")[-1]
                break

        logger.info("=" * 60)
        logger.info("Evaluating checkpoint: %s (step %s)", ckpt_path, step)
        logger.info("=" * 60)

        metrics = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            dataset=dataset,
            device=args.device,
            max_samples=args.max_samples,
            hf_repo=args.hf_repo,
            whisper_model=args.whisper_model,
        )

        # Save
        metrics_path = output_dir / f"step_{step}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"step": step, **metrics}, f, indent=2)

        # Print results
        print(f"\n{'='*50}")
        print(f"Step {step} — Big Bench Audio Results")
        print(f"{'='*50}")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
