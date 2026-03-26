"""End-to-end SteerDuplex data generation pipeline.

All phases are resumable — rerunning will skip completed work.

Usage:
    python run_pipeline.py                                    # full pipeline
    python run_pipeline.py --phase 1                          # transcript generation only
    python run_pipeline.py --from_phase 3 --to_phase 4       # TTS + assembly
    python run_pipeline.py --category A9_dynamic_steering     # specific category

Categories that require special assembly features not yet implemented are
excluded by default. Use --include_all to force-include them.
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
    5: ("Audio Split & Trim", "pipeline.split_audio"),
    6: ("System Prompt Rephrasing", "pipeline.rephrase_system_prompts"),
    7: ("External Data Prompting", "pipeline.add_prompts_external"),
    8: ("Dataset Formatting", "pipeline.format_dataset"),
}

# Categories that need assembly features not yet implemented:
# - B10.audio_cue_memory needs ambient_audio_injection (mixing environmental sounds)
# These are excluded from the default pipeline run. When the assembly pipeline
# gains these features, remove them from this set.
EXCLUDED_CATEGORIES: set[str] = {
    # B10.audio_cue_memory requires ambient sound mixing into user audio channel.
    # The subcategory is inside B10 so we can't exclude just the subcategory at
    # the pipeline level — B10 as a whole is fine, the LLM will just generate
    # audio_cue_memory transcripts that won't get ambient sounds mixed in.
    # This is acceptable: the TEXT content still teaches semantic association.
    # So nothing is fully excluded — all categories can run through the pipeline.
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
    parser.add_argument("--phase", type=int, default=None, help="Run single phase (1-8)")
    parser.add_argument("--from_phase", type=int, default=1)
    parser.add_argument("--to_phase", type=int, default=8)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--skip_whisper", action="store_true")
    parser.add_argument("--clean", action="store_true",
                        help="Remove old assembled/formatted data before re-running (use when assembly format changed)")
    parser.add_argument("--run_quality_filter", action="store_true",
                        help="Run quality filter between TTS and assembly (off by default)")
    parser.add_argument("--num_gpus", type=int, default=None, help="GPUs for TTS synthesis (default: all available)")
    parser.add_argument("--workers_per_gpu", type=int, default=None, help="Concurrent TTS workers per GPU (default: 4)")
    parser.add_argument("--scale", choices=["pilot", "full"], default="pilot",
                        help="Use pilot or full training conversation counts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    # Check if the requested category is excluded
    if args.category and args.category in EXCLUDED_CATEGORIES:
        print(f"[SKIP] Category {args.category} requires assembly features not yet implemented.")
        print(f"       Excluded categories: {sorted(EXCLUDED_CATEGORIES)}")
        sys.exit(0)

    print(f"SteerDuplex Pipeline | Config: {config_path}")

    if args.phase:
        phases = [args.phase]
    else:
        phases = list(range(args.from_phase, args.to_phase + 1))

    # Clean old data if requested (needed when assembly format changes)
    if args.clean and 4 in phases:
        import shutil
        cfg = load_yaml(config_path)
        assembled_dir = Path(cfg.get("assembly", {}).get("output_dir", "data/assembled"))
        formatted_dir = Path(cfg.get("dataset", {}).get("output_dir", "data/formatted"))
        for d in [assembled_dir, formatted_dir]:
            if d.exists():
                print(f"  Removing {d}/...")
                shutil.rmtree(d, ignore_errors=True)
                # Handle race with other processes: retry if dir still exists
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)

    for phase in phases:
        if phase not in PHASES:
            print(f"Unknown phase: {phase}")
            sys.exit(1)

        name, module = PHASES[phase]
        extra = ["--seed", str(args.seed)]
        if phase == 1:
            extra.extend(["--scale", args.scale])
        if phase == 3:
            if args.num_gpus:
                extra.extend(["--num_gpus", str(args.num_gpus)])
            if args.workers_per_gpu:
                extra.extend(["--workers_per_gpu", str(args.workers_per_gpu)])
        if phase == 5:
            # Audio split: 128 workers, distributed-safe
            extra = ["--num_workers", "128"]
        if phase == 6:
            # Rephrasing: use 96 workers by default
            extra = ["--num_workers", "96"]
        if phase == 7:
            # Import all registered external datasets first (creates audio/ + manifest)
            import_cmd = [sys.executable, "-m", "pipeline.import_external", "--dataset", "all"]
            print(f"\n  Pre-step: importing external datasets...")
            print(f"  {' '.join(import_cmd)}")
            subprocess.run(import_cmd)
            # Then add prompt regions with LLM-generated system prompts
            extra = ["--dataset", "all", "--use_llm", "--num_workers", "96"]
        if phase == 8 and args.skip_whisper:
            extra.append("--skip_whisper")

        if not run_phase(name, module, args.config, args.category, extra):
            sys.exit(1)

        # Quality filter is OFF by default — use all generated audio.
        # Enable with --run_quality_filter if you want WER-based filtering.
        if phase == 3 and args.run_quality_filter:
            from pipeline.quality_filter import main as _  # noqa: F401 — verify importable
            qf_cmd = [sys.executable, "-m", "pipeline.quality_filter", "--config", str(config_path)]
            if args.category:
                qf_cmd.extend(["--category", args.category])
            if args.num_gpus:
                qf_cmd.extend(["--num_gpus", str(args.num_gpus)])
            if args.workers_per_gpu:
                qf_cmd.extend(["--workers_per_gpu", str(args.workers_per_gpu)])
            print(f"\n{'=' * 60}")
            print(f"  Quality Filter (optional)")
            print(f"  {' '.join(qf_cmd)}")
            print(f"{'=' * 60}\n")
            subprocess.run(qf_cmd)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete!")
    print(f"{'=' * 60}")

    if 8 in phases:
        cfg = load_yaml(config_path)
        print(f"\nTrain manifest: {cfg['dataset']['output_dir']}/manifest_train.jsonl")
        print(f"Eval manifest:  {cfg['dataset']['output_dir']}/manifest_eval.jsonl")
        print(f"\nTo train:")
        print(f"  bash training/launch.sh configs/full_training.yaml 8")


if __name__ == "__main__":
    main()
