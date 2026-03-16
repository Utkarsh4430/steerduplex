"""Merge dataset manifests based on training config.

Called automatically by launch.sh before training starts.
Can also be run standalone to preview what will be merged.

Audio paths in each source manifest are relative to that manifest's directory.
When merging into a combined manifest at a different location, paths are
rewritten to be relative to the combined manifest's parent directory.

Standard dataset format:
    data/external/{name}/manifest.jsonl
    data/formatted/manifest_train.jsonl  (synthetic)

Usage:
    python -m pipeline.merge_manifests --config configs/full_training.yaml
    python -m pipeline.merge_manifests --config configs/full_training.yaml --dry_run
"""

import argparse
import json
import os
import random
from pathlib import Path

from pipeline.utils import load_yaml


def _rebase_paths(entries: list[dict], src_manifest: Path, dst_manifest: Path) -> list[dict]:
    """Rewrite relative audio paths so they resolve correctly from dst_manifest's directory.

    sphn.dataset_jsonl resolves paths relative to the JSONL file's parent directory.
    When merging manifests from different locations, we need to adjust paths.
    """
    src_dir = src_manifest.parent.resolve()
    dst_dir = dst_manifest.parent.resolve()

    if src_dir == dst_dir:
        return entries

    rebased = []
    for entry in entries:
        entry = dict(entry)  # shallow copy
        if "path" in entry:
            # Resolve the absolute path from the source manifest
            abs_path = src_dir / entry["path"]
            # Make it relative to the destination manifest directory
            try:
                entry["path"] = os.path.relpath(abs_path, dst_dir)
            except ValueError:
                # Different drives on Windows — use absolute path
                entry["path"] = str(abs_path)
        rebased.append(entry)
    return rebased


def load_manifest(path: Path, max_hours: float | None = None, seed: int = 42) -> list[dict]:
    if not path.exists():
        return []

    with open(path) as f:
        all_entries = [json.loads(line) for line in f]

    if max_hours is None:
        return all_entries

    # Shuffle before capping to avoid systematic bias toward early entries
    rng = random.Random(seed)
    rng.shuffle(all_entries)

    entries = []
    total_sec = 0.0
    max_sec = max_hours * 3600

    for entry in all_entries:
        dur = entry.get("duration", 0)
        if total_sec + dur > max_sec:
            break
        entries.append(entry)
        total_sec += dur

    return entries


def merge(config_path: str, dry_run: bool = False, seed: int = 42) -> dict:  # noqa: C901
    """Merge enabled datasets into combined train/eval manifests.

    Returns stats dict: {name: {entries, hours, enabled}}.
    """
    cfg = load_yaml(config_path)
    datasets = cfg.get("datasets", {})
    data_cfg = cfg.get("data", {})

    out_train = Path(data_cfg.get("train_data", "data/formatted/manifest_combined_train.jsonl"))
    out_eval = Path(data_cfg.get("eval_data", "data/formatted/manifest_combined_eval.jsonl"))

    all_train = []
    all_eval = []
    stats = {}

    print(f"{'Dataset':20s} {'Status':8s} {'Cap':>10s} {'Entries':>10s} {'Hours':>10s} {'Manifest'}")
    print("-" * 90)

    for name, ds in datasets.items():
        enabled = ds.get("enabled", False)
        max_hours_raw = ds.get("max_hours", "all")
        max_hours = None if max_hours_raw == "all" else float(max_hours_raw)

        if not enabled:
            stats[name] = {"entries": 0, "hours": 0.0, "enabled": False}
            print(f"{name:20s} {'OFF':8s} {'—':>10s} {'—':>10s} {'—':>10s}")
            continue

        # Load train manifest
        train_path = Path(ds.get("manifest_train", ""))
        entries = load_manifest(train_path, max_hours, seed) if train_path.name else []
        hours = sum(e.get("duration", 0) for e in entries) / 3600
        cap_str = "all" if max_hours is None else f"{max_hours:.0f}h"

        if not entries and train_path.name:
            status = "MISSING"
        else:
            status = "ON"
            # Rebase paths relative to the output manifest location
            entries = _rebase_paths(entries, train_path, out_train)
            all_train.extend(entries)

        stats[name] = {"entries": len(entries), "hours": round(hours, 1), "enabled": True}
        print(f"{name:20s} {status:8s} {cap_str:>10s} {len(entries):>10,d} {hours:>10.1f} {train_path}")

        # Eval manifest (optional)
        eval_path_str = ds.get("manifest_eval")
        if eval_path_str:
            eval_path = Path(eval_path_str)
            eval_entries = load_manifest(eval_path)
            eval_entries = _rebase_paths(eval_entries, eval_path, out_eval)
            all_eval.extend(eval_entries)

    total_hours = sum(s["hours"] for s in stats.values())
    total_entries = sum(s["entries"] for s in stats.values())
    print("-" * 90)
    print(f"{'TOTAL':20s} {'':8s} {'':>10s} {total_entries:>10,d} {total_hours:>10.1f}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return stats

    if not all_train:
        print("\n[ERROR] No training data found.")
        return stats

    # Shuffle and write
    random.seed(seed)
    random.shuffle(all_train)

    # Auto-split eval if none provided
    if not all_eval:
        n_eval = max(1, int(len(all_train) * 0.05))
        all_eval = all_train[:n_eval]
        all_train = all_train[n_eval:]
        print(f"\nNo eval manifests — auto-split 5% ({len(all_eval)} entries)")

    out_train.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w") as f:
        for entry in all_train:
            f.write(json.dumps(entry) + "\n")
    with open(out_eval, "w") as f:
        for entry in all_eval:
            f.write(json.dumps(entry) + "\n")

    print(f"\nTrain: {out_train} ({len(all_train):,d} entries)")
    print(f"Eval:  {out_eval} ({len(all_eval):,d} entries)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge dataset manifests")
    parser.add_argument("--config", type=str, default="configs/full_training.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Preview only, don't write")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    merge(args.config, args.dry_run, args.seed)


if __name__ == "__main__":
    main()
