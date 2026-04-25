"""Phase 2a of VoiceBench evaluation: local Parakeet ASR.

Runs NVIDIA NeMo's parakeet-tdt-0.6b-v2 model over every Moshi output audio and
writes the plain-text transcription back into each split's output.json.

Designed to be invoked as a subprocess from the main pipeline (llm_judge.py),
typically in a dedicated conda env that has nemo_toolkit installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from eval.voicebench.splits import (
    SDQA_DEFAULT_REGIONS,
    SUPPORTED_SUBSETS,
    build_specs,
)

logger = logging.getLogger(__name__)

PARAKEET_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(results, fh, indent=2)
    os.replace(tmp, path)


def is_invalid_transcription(value: Any) -> bool:
    """True for records that need (re)transcription.

    Catches both missing values and the old Whisper-verbose-JSON blobs the
    previous pipeline stored as strings like `{"text": "...", "segments": ...}`.
    """
    if value is None:
        return True
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if stripped.startswith("{"):
        return True
    return False


def _extract_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if text is None:
        text = str(result)
    return text.strip()


def transcribe_split(
    split_dir: Path,
    asr_model,
    batch_size: int,
    force: bool,
) -> None:
    """Fill `transcription` for every record in split_dir/output.json."""
    output_json = split_dir / "output.json"
    if not output_json.exists():
        logger.warning("[%s] no output.json; skipping.", split_dir.name)
        return

    results = load_results(output_json)

    if force:
        for r in results:
            r["transcription"] = None
            r["score"] = None

    pending: List[Dict[str, Any]] = []
    for r in results:
        if is_invalid_transcription(r.get("transcription")):
            # Invalidate stale downstream scores whenever we re-transcribe — a score
            # computed against a JSON blob is not salvageable.
            if r.get("transcription") is not None:
                r["transcription"] = None
                r["score"] = None
            pending.append(r)

    if not pending:
        logger.info("[%s] all transcriptions valid; skipping Parakeet.", split_dir.name)
        return

    logger.info(
        "[%s] transcribing %d audio files with Parakeet (batch=%d).",
        split_dir.name, len(pending), batch_size,
    )

    total = len(pending)
    with tqdm(total=total, desc=f"{split_dir.name} parakeet",
              unit="file", dynamic_ncols=True) as pbar:
        for start in range(0, total, batch_size):
            batch = pending[start:start + batch_size]
            abs_paths = [str(split_dir / entry["output_audio_path"]) for entry in batch]
            try:
                outputs = asr_model.transcribe(abs_paths, batch_size=len(abs_paths))
                texts = [_extract_text(o) for o in outputs]
            except Exception:
                logger.exception("[%s] batch at offset %d failed; marking empty.",
                                 split_dir.name, start)
                texts = [""] * len(batch)

            for entry, text in zip(batch, texts):
                entry["transcription"] = text

            pbar.update(len(batch))
            save_results(results, output_json)


def load_parakeet(asr_device: str):
    import nemo.collections.asr as nemo_asr
    import torch

    logger.info("Loading %s onto %s...", PARAKEET_MODEL_NAME, asr_device)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL_NAME)
    model = model.to(torch.device(asr_device))
    model.eval()
    return model


def _needs_any_work(output_dir: Path, specs, force: bool) -> bool:
    if force:
        return True
    for spec in specs:
        p = output_dir / spec.name / "output.json"
        if not p.exists():
            continue
        try:
            records = load_results(p)
        except Exception:
            continue
        if any(is_invalid_transcription(r.get("transcription")) for r in records):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceBench Phase 2a (Parakeet ASR)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=list(SUPPORTED_SUBSETS))
    parser.add_argument("--sdqa_regions", nargs="+", default=None)
    parser.add_argument("--asr_device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force", action="store_true",
                        help="Wipe transcription+score fields and re-transcribe everything.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    output_dir = Path(args.output_dir)

    sdqa_regions = args.sdqa_regions
    if "sd-qa" in args.splits and sdqa_regions is None:
        sdqa_regions = list(SDQA_DEFAULT_REGIONS)
    specs = build_specs(args.splits, sdqa_regions or [])

    if not _needs_any_work(output_dir, specs, args.force):
        logger.info("All splits already have valid transcriptions; nothing to do.")
        return

    asr_model = load_parakeet(args.asr_device)

    for spec in specs:
        split_dir = output_dir / spec.name
        transcribe_split(split_dir, asr_model, args.batch_size, args.force)


if __name__ == "__main__":
    main()
