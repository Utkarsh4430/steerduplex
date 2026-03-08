"""Shared utilities for the SteerDuplex pipeline."""

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(data: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
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
    """Get duration of audio file in seconds."""
    import soundfile as sf
    info = sf.info(str(path))
    return info.duration


def load_all_categories(categories_dir: str | Path) -> dict[str, dict]:
    """Load all A1-A8 YAML category files."""
    categories = {}
    for p in sorted(Path(categories_dir).glob("A*.yaml")):
        cat_id = p.stem  # e.g. "A1_customer_service"
        categories[cat_id] = load_yaml(p)
    return categories
