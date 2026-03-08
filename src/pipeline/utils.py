"""Shared utilities for the SteerDuplex pipeline."""

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path):
    """Atomic JSON save — writes to temp file then renames to prevent corruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(data: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    if not Path(path).exists():
        return items
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_audio_duration(path: str | Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.duration


def load_all_categories(categories_dir: str | Path) -> dict[str, dict]:
    """Load all A1-A10 YAML category files."""
    categories = {}
    for p in sorted(Path(categories_dir).glob("A*.yaml")):
        cat_id = p.stem
        categories[cat_id] = load_yaml(p)
    return categories


def get_completed_ids(output_dir: str | Path, suffix: str = ".json") -> set[str]:
    """Get set of already-completed conversation IDs in a directory (for resumability)."""
    completed = set()
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return completed
    for p in output_dir.rglob(f"*{suffix}"):
        completed.add(p.stem)
    return completed
