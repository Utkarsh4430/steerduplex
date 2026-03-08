"""End-to-end SteerDuplex data generation pipeline.

All phases are resumable — rerunning will skip completed work.

Usage:
    python run_pipeline.py                                    # full pipeline
    python run_pipeline.py --phase 1                          # transcript generation only
    python run_pipeline.py --from_phase 3 --to_phase 4       # TTS + assembly
    python run_pipeline.py --category A9_dynamic_steering     # specific category
"""

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline.utils import load_yaml


PHASES = {
    1: ("Transcript Generation", "pipeline.generate_transcripts"),
    2: ("Voice Assignment", "pipeline.assign_voices"),
    3: ("TTS Synthesis", "pipeline.synthesize_tts"),
    4: ("Channel Assembly", "pipeline.assemble_channels"),
    5: ("Dataset Formatting", "pipeline.format_dataset"),
}

# Optional phases run between main phases
OPTIONAL_PHASES = {
    "quality_filter": ("Quality Filter", "pipeline.quality_filter"),
}


def run_phase(phase_name: str, module: str, config: str, category: str | None = None, extra_args: list[str] | None = None) -> bool:
    cmd = [sys.executable, "-m", module, "--config", config]
    if category:
        cmd.extend(["--category", category])
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'=' * 60}")
    print(f"  {phase_name}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] {phase_name} failed (exit {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="SteerDuplex data generation pipeline")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--phase", type=int, default=None, help="Run single phase (1-5)")
    parser.add_argument("--from_phase", type=int, default=1)
    parser.add_argument("--to_phase", type=int, default=5)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--skip_whisper", action="store_true")
    parser.add_argument("--skip_quality", action="store_true", help="Skip quality filter between TTS and assembly")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    print(f"SteerDuplex Pipeline | Config: {config_path}")

    if args.phase:
        phases = [args.phase]
    else:
        phases = list(range(args.from_phase, args.to_phase + 1))

    for phase in phases:
        if phase not in PHASES:
            print(f"Unknown phase: {phase}")
            sys.exit(1)

        name, module = PHASES[phase]
        extra = ["--seed", str(args.seed)]
        if phase == 5 and args.skip_whisper:
            extra.append("--skip_whisper")

        if not run_phase(name, module, args.config, args.category, extra):
            sys.exit(1)

        # Run quality filter after TTS (phase 3), before assembly (phase 4)
        if phase == 3 and not args.skip_quality and 4 in phases:
            qf_name, qf_module = OPTIONAL_PHASES["quality_filter"]
            run_phase(qf_name, qf_module, args.config, args.category)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete!")
    print(f"{'=' * 60}")

    if 5 in phases:
        cfg = load_yaml(config_path)
        print(f"\nTrain manifest: {cfg['dataset']['output_dir']}/manifest_train.jsonl")
        print(f"Eval manifest:  {cfg['dataset']['output_dir']}/manifest_eval.jsonl")
        print(f"\nTo train:")
        print(f"  bash training/launch.sh configs/pilot_training.yaml 8")


if __name__ == "__main__":
    main()
