"""End-to-end SteerDuplex data generation pipeline.

Orchestrates all phases:
  Phase 1: Generate conversation transcripts (LLM)
  Phase 2: Assign voices
  Phase 3: Synthesize TTS audio
  Phase 4: Assemble 2-channel stereo
  Phase 5: Format for moshi-finetune

Usage:
    # Run full pipeline
    python run_pipeline.py --config configs/generation.yaml

    # Run specific phase
    python run_pipeline.py --config configs/generation.yaml --phase 1

    # Run from phase 3 onward (skip transcript generation and voice assignment)
    python run_pipeline.py --config configs/generation.yaml --from_phase 3

    # Specific category only
    python run_pipeline.py --config configs/generation.yaml --category A3_tone_controlled
"""

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline.utils import load_yaml


def run_phase(phase: int, config: str, category: str | None = None, extra_args: list[str] | None = None):
    """Run a specific pipeline phase."""
    modules = {
        1: "pipeline.generate_transcripts",
        2: "pipeline.assign_voices",
        3: "pipeline.synthesize_tts",
        4: "pipeline.assemble_channels",
        5: "pipeline.format_dataset",
    }

    module = modules.get(phase)
    if not module:
        print(f"Unknown phase: {phase}")
        return False

    cmd = [sys.executable, "-m", module, "--config", config]
    if category:
        cmd.extend(["--category", category])
    if extra_args:
        cmd.extend(extra_args)

    phase_names = {
        1: "Transcript Generation",
        2: "Voice Assignment",
        3: "TTS Synthesis",
        4: "Channel Assembly",
        5: "Dataset Formatting",
    }

    print(f"\n{'='*60}")
    print(f"  Phase {phase}: {phase_names[phase]}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Phase {phase} failed with exit code {result.returncode}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="SteerDuplex data generation pipeline")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--phase", type=int, default=None, help="Run specific phase (1-5)")
    parser.add_argument("--from_phase", type=int, default=1, help="Start from this phase")
    parser.add_argument("--to_phase", type=int, default=5, help="End at this phase (inclusive)")
    parser.add_argument("--category", type=str, default=None, help="Specific category")
    parser.add_argument("--skip_whisper", action="store_true", help="Skip Whisper in Phase 5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = load_yaml(config_path)
    print(f"SteerDuplex Pipeline")
    print(f"Config: {config_path}")
    print(f"Pilot target: {cfg['transcript']['pilot_per_category']} convos per category")

    if args.phase:
        phases = [args.phase]
    else:
        phases = list(range(args.from_phase, args.to_phase + 1))

    for phase in phases:
        extra = []
        if phase == 5 and args.skip_whisper:
            extra.append("--skip_whisper")

        success = run_phase(phase, args.config, args.category, extra)
        if not success:
            print(f"\nPipeline stopped at Phase {phase}.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"{'='*60}")

    if 5 in phases:
        manifest = cfg["dataset"]["manifest_path"]
        print(f"\nDataset manifest: {manifest}")
        print(f"\nTo train:")
        print(f"  bash training/launch.sh configs/pilot_training.yaml <num_gpus>")


if __name__ == "__main__":
    main()
