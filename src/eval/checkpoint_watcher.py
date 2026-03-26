"""Checkpoint watcher: runs FD-Bench v1.0 evaluation on new checkpoints during training.

Monitors a training run directory for new checkpoints, runs FD-Bench v1.0
evaluation on each, and logs results to Weights & Biases in the same project
as the training run.

Launch alongside training:
    # In a separate terminal / tmux pane (while training runs on 8 GPUs):
    python -m eval.checkpoint_watcher \
        --run_dir runs/full_v3_20260320_101535 \
        --data_dir data/benchmarks/fd_bench_v1 \
        --device cuda:0

    # With sampling for faster eval (~10min per checkpoint):
    python -m eval.checkpoint_watcher \
        --run_dir runs/full_v3_20260320_101535 \
        --data_dir data/benchmarks/fd_bench_v1 \
        --max_samples 50 \
        --device cuda:0

The watcher creates a separate wandb run grouped with the training run,
so benchmark metrics appear alongside training loss in the dashboard.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def find_checkpoints(run_dir: Path) -> list[dict]:
    """Find all valid checkpoints in a run directory.

    Returns sorted list of {"step": int, "path": str, "name": str}.
    Handles both 'checkpoint_XXXXXX' and 'step_XXXXXX' naming conventions.
    """
    ckpt_base = run_dir / "checkpoints"
    if not ckpt_base.exists():
        return []

    checkpoints = []
    for pattern in ["checkpoint_*", "step_*"]:
        for d in ckpt_base.glob(pattern):
            if not d.is_dir():
                continue
            # Look for consolidated weights
            consolidated = d / "consolidated" / "consolidated.safetensors"
            if not consolidated.exists():
                continue

            # Extract step number from directory name
            parts = d.name.split("_")
            try:
                step = int(parts[-1])
            except (ValueError, IndexError):
                logger.warning("Cannot parse step from %s — skipping", d.name)
                continue

            checkpoints.append({
                "step": step,
                "path": str(d / "consolidated"),
                "name": d.name,
            })

    return sorted(checkpoints, key=lambda x: x["step"])


def get_run_name(run_dir: Path) -> str:
    """Extract run name from directory."""
    return run_dir.name


def get_max_steps(run_dir: Path) -> int | None:
    """Read max_steps from the saved args.yaml."""
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return None
    with open(args_path) as f:
        args = yaml.safe_load(f)
    return args.get("max_steps")


def get_last_completed_step(run_dir: Path) -> int | None:
    """Read the last completed training step from resume_state.pt."""
    resume_path = run_dir / "resume_state.pt"
    if not resume_path.exists():
        return None
    try:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        return state.get("step")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------
def watch(
    run_dir: str,
    data_dir: str,
    output_dir: str = "eval_outputs/fd_bench_v1",
    device: str = "cuda:0",
    poll_interval: int = 120,
    max_samples: int | None = 50,
    system_prompt: str = "You enjoy having a good conversation.",
    voice_prompt: str | None = None,
    whisper_model: str = "large-v3",
    hf_repo: str = "kyutai/moshiko-pytorch-bf16",
    wandb_project: str = "steerduplex",
    skip_existing: bool = False,
    inference_devices: list[str] | None = None,
    workers_per_gpu: int = 1,
):
    """Watch for new checkpoints and run FD-Bench v1.0 evaluation.

    Args:
        run_dir: Training run directory (e.g., runs/full_v3_20260320_101535).
        data_dir: Path to FD-Bench v1.0 dataset.
        output_dir: Where to save inference outputs and ASR results.
        device: CUDA device for model inference and ASR.
        poll_interval: Seconds between checkpoint polls.
        max_samples: Max samples per task (None = all, 50 recommended for speed).
        system_prompt: System prompt for the model during eval.
        voice_prompt: Optional voice prompt WAV path.
        whisper_model: Whisper model size for ASR.
        hf_repo: HuggingFace repo for base model weights.
        wandb_project: Weights & Biases project name.
        skip_existing: If True, skip checkpoints that already exist at startup.
    """
    import wandb
    from eval.fd_bench_v1 import FDBenchV1Evaluator

    run_dir = Path(run_dir)
    if not run_dir.exists():
        logger.info("Run dir %s does not exist yet — waiting...", run_dir)
        while not run_dir.exists():
            time.sleep(poll_interval)
        logger.info("Run dir appeared: %s", run_dir)

    run_name = get_run_name(run_dir)

    # Initialize wandb — separate run, grouped with training
    wandb.init(
        project=wandb_project,
        name=f"{run_name}_fdBench",
        group=run_name,
        job_type="eval",
        tags=["fd_bench_v1", "eval", "benchmark"],
        config={
            "eval_type": "fd_bench_v1",
            "training_run": run_name,
            "training_run_dir": str(run_dir),
            "max_samples_per_task": max_samples,
            "system_prompt": system_prompt,
            "whisper_model": whisper_model,
            "device": device,
        },
    )

    # Log baseline reference values from FD-Bench v1.0 leaderboard
    # These appear as horizontal lines on wandb charts for comparison.
    # Source: https://github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v1_v1.5
    _BASELINES = {
        "PersonaPlex": {
            "fd_bench/pause_handling/synthetic_tor": 0.584,
            "fd_bench/pause_handling/candor_tor": 0.662,
            "fd_bench/backchannel/tor": 0.327,
            "fd_bench/backchannel/frequency": 0.025,
            "fd_bench/backchannel/jsd": 0.649,
            "fd_bench/turn_taking/candor_tor": 0.992,
            "fd_bench/turn_taking/latency": 0.070,
            "fd_bench/user_interruption/tor": 1.000,
            "fd_bench/user_interruption/gpt_score": 4.210,
            "fd_bench/user_interruption/latency": 0.400,
        },
        "GPT-Realtime": {
            "fd_bench/pause_handling/synthetic_tor": 0.010,
            "fd_bench/pause_handling/candor_tor": 0.120,
            "fd_bench/backchannel/tor": 0.000,
            "fd_bench/backchannel/frequency": 0.007,
            "fd_bench/backchannel/jsd": 0.980,
            "fd_bench/turn_taking/candor_tor": 1.000,
            "fd_bench/turn_taking/latency": 1.470,
            "fd_bench/user_interruption/tor": 0.970,
            "fd_bench/user_interruption/gpt_score": 3.850,
            "fd_bench/user_interruption/latency": 1.500,
        },
        "Moshi-base": {
            "fd_bench/pause_handling/synthetic_tor": 0.985,
            "fd_bench/pause_handling/candor_tor": 0.980,
            "fd_bench/backchannel/tor": 1.000,
            "fd_bench/backchannel/frequency": 0.001,
            "fd_bench/backchannel/jsd": 0.957,
            "fd_bench/turn_taking/candor_tor": 0.941,
            "fd_bench/turn_taking/latency": 0.265,
            "fd_bench/user_interruption/tor": 1.000,
            "fd_bench/user_interruption/gpt_score": 0.765,
            "fd_bench/user_interruption/latency": 0.257,
        },
    }
    logger.info("Baselines loaded: %s", list(_BASELINES.keys()))

    evaluator = FDBenchV1Evaluator(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        system_prompt=system_prompt,
        voice_prompt=voice_prompt,
        max_samples_per_task=max_samples,
        whisper_model=whisper_model,
        hf_repo=hf_repo,
        inference_devices=inference_devices,
        workers_per_gpu=workers_per_gpu,
    )

    evaluated_steps: set[int] = set()

    # Optionally skip existing checkpoints (useful when restarting the watcher)
    if skip_existing:
        existing = find_checkpoints(run_dir)
        for ckpt in existing:
            evaluated_steps.add(ckpt["step"])
        if evaluated_steps:
            logger.info(
                "Skipping %d existing checkpoints (steps: %s)",
                len(evaluated_steps),
                sorted(evaluated_steps),
            )

    logger.info(
        "Watching %s for new checkpoints (poll every %ds, max %s samples/task)...",
        run_dir, poll_interval,
        max_samples if max_samples else "all",
    )

    while True:
        checkpoints = find_checkpoints(run_dir)
        new_ckpts = [c for c in checkpoints if c["step"] not in evaluated_steps]

        for ckpt in new_ckpts:
            step = ckpt["step"]
            logger.info("=" * 60)
            logger.info("New checkpoint: %s (step %d)", ckpt["name"], step)
            logger.info("=" * 60)

            try:
                metrics = evaluator.evaluate(ckpt["path"], step=step)

                # Log our metrics
                wandb.log(metrics, step=step)

                # Log a comparison summary table at each step
                _metric_keys = [
                    ("Pause: Synth TOR", "fd_bench/pause_handling/synthetic_tor", "↓"),
                    ("Pause: Candor TOR", "fd_bench/pause_handling/candor_tor", "↓"),
                    ("Backchannel TOR", "fd_bench/backchannel/tor", "↓"),
                    ("Backchannel Freq", "fd_bench/backchannel/frequency", "↑"),
                    ("Turn-Taking TOR", "fd_bench/turn_taking/candor_tor", "↑"),
                    ("TT Latency", "fd_bench/turn_taking/latency", "↓"),
                    ("Interrupt TOR", "fd_bench/user_interruption/tor", "↑"),
                    ("Interrupt GPT", "fd_bench/user_interruption/gpt_score", "↑"),
                    ("Interrupt Latency", "fd_bench/user_interruption/latency", "↓"),
                ]
                table_data = []
                for name, key, direction in _metric_keys:
                    ours = metrics.get(key)
                    pp = _BASELINES["PersonaPlex"].get(key)
                    moshi = _BASELINES["Moshi-base"].get(key)
                    gpt_rt = _BASELINES["GPT-Realtime"].get(key)
                    if ours is not None:
                        table_data.append([name, direction, round(ours, 3),
                                           pp, gpt_rt, moshi])
                if table_data:
                    comparison_table = wandb.Table(
                        columns=["Metric", "Dir", "Ours", "PersonaPlex", "GPT-RT", "Moshi"],
                        data=table_data,
                    )
                    wandb.log({"comparison_table": comparison_table}, step=step)

                # Also save metrics locally
                metrics_path = Path(output_dir) / f"step_{step}" / "metrics.json"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_path, "w") as f:
                    json.dump({"step": step, **metrics}, f, indent=2)

                logger.info("Step %d — logged %d metrics to wandb", step, len(metrics))
                evaluated_steps.add(step)

            except Exception as e:
                logger.error(
                    "Evaluation failed for step %d: %s", step, e, exc_info=True,
                )
                # Mark as evaluated to avoid infinite retries
                evaluated_steps.add(step)

        # Check if training is complete
        max_steps = get_max_steps(run_dir)
        last_step = get_last_completed_step(run_dir)
        if (
            max_steps is not None
            and last_step is not None
            and last_step >= max_steps
            and last_step in evaluated_steps
        ):
            logger.info(
                "Training complete (step %d >= max_steps %d) and final "
                "checkpoint evaluated. Exiting.",
                last_step, max_steps,
            )
            break

        time.sleep(poll_interval)

    wandb.finish()
    logger.info("Checkpoint watcher finished.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FD-Bench v1.0 checkpoint watcher — run during training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (alongside training)
  python -m eval.checkpoint_watcher \\
      --run_dir runs/full_v3_20260320_101535 \\
      --data_dir data/benchmarks/fd_bench_v1

  # Faster eval with 50 samples per task
  python -m eval.checkpoint_watcher \\
      --run_dir runs/full_v3_20260320_101535 \\
      --data_dir data/benchmarks/fd_bench_v1 \\
      --max_samples 50

  # Skip already-evaluated checkpoints (watcher restart)
  python -m eval.checkpoint_watcher \\
      --run_dir runs/full_v3_20260320_101535 \\
      --data_dir data/benchmarks/fd_bench_v1 \\
      --skip_existing
        """,
    )
    parser.add_argument(
        "--run_dir", required=True,
        help="Training run directory (e.g., runs/full_v3_20260320_101535)",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="FD-Bench v1.0 dataset root directory",
    )
    parser.add_argument("--output_dir", default="eval_outputs/fd_bench_v1")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device for inference + ASR")
    parser.add_argument("--poll_interval", type=int, default=120,
                        help="Seconds between polls (default: 120)")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Max samples per task (default: 50, use 0 for all)")
    parser.add_argument("--system_prompt",
                        default="You enjoy having a good conversation.",
                        help="System prompt for model during eval")
    parser.add_argument("--voice_prompt", default=None,
                        help="Optional voice prompt WAV path")
    parser.add_argument("--whisper_model", default="large-v3",
                        help="Whisper model size for ASR")
    parser.add_argument("--hf_repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--wandb_project", default="steerduplex")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip checkpoints that exist at startup")
    parser.add_argument("--inference_devices", nargs="+", default=None,
                        help="GPU devices for parallel inference (e.g. cuda:1 cuda:2 cuda:3)")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Parallel workers per GPU (default: 1, use 3 for 80GB GPUs)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    watch(
        run_dir=args.run_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        poll_interval=args.poll_interval,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        system_prompt=args.system_prompt,
        voice_prompt=args.voice_prompt,
        whisper_model=args.whisper_model,
        hf_repo=args.hf_repo,
        wandb_project=args.wandb_project,
        skip_existing=args.skip_existing,
        inference_devices=args.inference_devices,
        workers_per_gpu=args.workers_per_gpu,
    )


if __name__ == "__main__":
    main()
