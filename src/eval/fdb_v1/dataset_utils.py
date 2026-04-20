"""Sample discovery and mirror-tree construction for Full Duplex Bench v1/v1.5.

FDB eval scripts glob `{root_dir}/*/...` expecting input.wav + output.wav +
annotation JSONs co-located in each sample folder. We symlink (or copy) the
read-only dataset inputs + annotations into a user-supplied output root, then
write our generated output.wav / JSONs in-place next to the symlinks.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


# ---- Task registries ---------------------------------------------------------
#
# FDB v1.0's downloaded dataset splits each task across source-prefixed folders
# (e.g. pause_handling = candor_pause_handling + synthetic_pause_handling).
# FDB's official eval scripts expect all samples for a task in ONE folder, so
# we union the sources into a single mirror dir per canonical task name.
#
# v1.5's folders match canonical task names directly.

TASKS_V1: Dict[str, Dict] = {
    "pause_handling": {
        "sources": ["candor_pause_handling", "synthetic_pause_handling"],
        "annotations": ["pause.json"],
        "optional_annotations": ["transcription.json"],
        "paired": False,
        "asr_task": "default",
    },
    "backchannel": {
        "sources": ["icc_backchannel"],
        "annotations": [],
        "optional_annotations": ["transcription.json"],
        "paired": False,
        "asr_task": "default",
    },
    "smooth_turn_taking": {
        "sources": ["candor_turn_taking"],
        "annotations": ["turn_taking.json"],
        "optional_annotations": ["transcription.json"],
        "paired": False,
        "asr_task": "default",
    },
    "user_interruption": {
        "sources": ["synthetic_user_interruption"],
        "annotations": ["interrupt.json"],
        "optional_annotations": ["context.wav", "interrupt.wav"],
        "paired": False,
        "asr_task": "user_interruption",
    },
}

# v1.5 zips ship pre-aligned `input.json` / `clean_input.json` (Parakeet word
# timings). We carry these through as optional annotations so eval_behavior.py
# reads the dataset's own transcripts rather than ASR re-runs.
_V15_OPTIONAL_INPUT_JSON = ["input.json", "clean_input.json"]

TASKS_V15: Dict[str, Dict] = {
    "user_interruption": {
        "sources": ["user_interruption"],
        "annotations": ["metadata.json"],
        "optional_annotations": list(_V15_OPTIONAL_INPUT_JSON),
        "paired": True,
    },
    "user_backchannel": {
        "sources": ["user_backchannel"],
        "annotations": ["metadata.json"],
        "optional_annotations": list(_V15_OPTIONAL_INPUT_JSON),
        "paired": True,
    },
    "talking_to_other": {
        "sources": ["talking_to_other"],
        "annotations": ["metadata.json"],
        "optional_annotations": list(_V15_OPTIONAL_INPUT_JSON),
        "paired": True,
    },
    "background_speech": {
        "sources": ["background_speech"],
        "annotations": ["metadata.json"],
        "optional_annotations": list(_V15_OPTIONAL_INPUT_JSON),
        "paired": True,
    },
}


def tasks_for_version(version: str) -> Dict[str, Dict]:
    if version == "1.0":
        return TASKS_V1
    if version == "1.5":
        return TASKS_V15
    if version == "both":
        return {**{f"v1_{k}": v for k, v in TASKS_V1.items()},
                **{f"v15_{k}": v for k, v in TASKS_V15.items()}}
    raise ValueError(f"Unknown version {version!r}; expected '1.0', '1.5', or 'both'")


# ---- Sample discovery --------------------------------------------------------

def _canonical_task_name(folder_name: str) -> str:
    """Strip `_example` suffix used in shipped example_data/ folders."""
    if folder_name.endswith("_example"):
        return folder_name[: -len("_example")]
    return folder_name


def discover_sample_dirs(task_dataset_dir: Path) -> List[Path]:
    """Return sample subdirs (each containing input.wav) under a task dir."""
    if not task_dataset_dir.is_dir():
        return []
    samples = []
    for child in sorted(task_dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if (child / "input.wav").exists():
            samples.append(child)
    return samples


# v1.5 sources are vendored locally (zips live next to the code, extracted on
# first run) so the pipeline is self-contained even when DATASET_ROOT only
# carries v1.0 sources.
_LOCAL_V15_DIR = Path(__file__).resolve().parent / "data" / "v1.5"


def _find_source_dir(dataset_root: Path, source: str, version_prefix: str) -> Path | None:
    """Locate a source directory under `dataset_root` (or vendored data/ for v1.5).

    Tries these candidates in order:
      - {dataset_root}/{source}
      - {dataset_root}/v1.0/{source}             (v1.0 only)
      - {dataset_root}/v1.5/{source}             (v1.5 only)
      - {pkg_dir}/data/v1.5/{source}             (v1.5 vendored)
      - {dataset_root}/{source}_example          (shipped example_data/ naming)

    Returns None if none exist.
    """
    candidates = [dataset_root / source]
    if version_prefix == "v1":
        candidates.append(dataset_root / "v1.0" / source)
    else:  # v15
        candidates.append(dataset_root / "v1.5" / source)
        candidates.append(_LOCAL_V15_DIR / source)
    candidates.append(dataset_root / f"{source}_example")
    for c in candidates:
        if c.is_dir() and any(discover_sample_dirs(c)):
            return c
    return None


def _source_prefix(source_dir_name: str, task: str) -> str:
    """Derive a short prefix to disambiguate samples that share an ID across sources.

    Rule: if source == task (v1.5 case) -> no prefix. Otherwise take the first
    underscore-separated token of the source dir name. Examples:
      candor_pause_handling, pause_handling -> 'candor'
      synthetic_user_interruption, user_interruption -> 'synthetic'
      candor_turn_taking, smooth_turn_taking -> 'candor'
      icc_backchannel, backchannel -> 'icc'
      user_interruption, user_interruption -> '' (v1.5, no prefix needed)
    """
    if source_dir_name == task:
        return ""
    return source_dir_name.split("_", 1)[0]


# ---- Mirror construction -----------------------------------------------------

def _link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def build_mirror_for_task(
    dataset_root: Path,
    output_root: Path,
    task: str,
    task_meta: Dict,
    use_copy: bool = False,
    out_subdir: str | None = None,
    version_prefix: str = "v1",
) -> List[Path]:
    """Mirror all source dirs for a task into `{output_root}/{out_subdir or task}/`.

    Sample IDs are prefixed with a short source token when the task has
    multiple sources (e.g. `candor_1`, `synthetic_1`), keeping every sample
    in a single mirror dir for FDB eval's `glob(root/*/...)`.

    Returns list of mirrored sample dirs (under output_root).
    """
    target_subdir = out_subdir or task
    paired = bool(task_meta.get("paired", False))
    annotations = list(task_meta.get("annotations", []))
    optional_annotations = list(task_meta.get("optional_annotations", []))
    sources = list(task_meta.get("sources", [task]))

    # Only prefix sample IDs when a task unions multiple sources. Single-source
    # tasks must keep the original sample names, since FDB's eval_backchannel.py
    # filters with `spk.isdigit()` and looks up `gt_distribution[spk]` by the
    # bare ICC speaker ID — `icc_<n>` would silently skip every sample.
    multi_source = len(sources) > 1

    mirrored: List[Path] = []
    for source in sources:
        source_dir = _find_source_dir(dataset_root, source, version_prefix)
        if source_dir is None:
            logger.warning("Source dir for %s not found under %s", source, dataset_root)
            continue
        prefix = _source_prefix(source, task) if multi_source else ""

        for sample_dir in discover_sample_dirs(source_dir):
            sample_name = f"{prefix}_{sample_dir.name}" if prefix else sample_dir.name
            out_sample_dir = output_root / target_subdir / sample_name
            out_sample_dir.mkdir(parents=True, exist_ok=True)

            _link_or_copy(sample_dir / "input.wav", out_sample_dir / "input.wav", use_copy)
            if paired:
                clean_src = sample_dir / "clean_input.wav"
                if clean_src.exists():
                    _link_or_copy(clean_src, out_sample_dir / "clean_input.wav", use_copy)

            for ann in annotations:
                src = sample_dir / ann
                if not src.exists():
                    logger.warning("Missing required annotation %s in %s", ann, sample_dir)
                    continue
                _link_or_copy(src, out_sample_dir / ann, use_copy)

            for ann in optional_annotations:
                src = sample_dir / ann
                if src.exists():
                    _link_or_copy(src, out_sample_dir / ann, use_copy)

            mirrored.append(out_sample_dir)

    return mirrored


# ---- Resume helpers ----------------------------------------------------------

def _wav_duration_seconds(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    if info.samplerate == 0:
        return 0.0
    return info.frames / info.samplerate


def inference_done_for_sample(sample_dir: Path, paired: bool, min_duration: float = 0.1) -> bool:
    """Return True iff inference outputs already exist and are non-empty.

    Does NOT enforce exact duration match against input — MoshiInference.generate()
    already pads/trims to input length. We just guard against truncated/empty
    write (e.g. a killed worker).
    """
    out = sample_dir / "output.wav"
    if not out.exists():
        return False
    try:
        if _wav_duration_seconds(out) < min_duration:
            return False
    except Exception:
        return False
    if paired:
        clean_src = sample_dir / "clean_input.wav"
        clean_out = sample_dir / "clean_output.wav"
        if clean_src.exists():
            if not clean_out.exists():
                return False
            try:
                if _wav_duration_seconds(clean_out) < min_duration:
                    return False
            except Exception:
                return False
    return True


# ---- Version utilities -------------------------------------------------------

def mirror_subdir_name(version_prefix: str, task: str) -> str:
    """Construct the subdir name under output_root for a given version+task.

    v1.0 user_interruption is disambiguated from v1.5 user_interruption by
    a `_v1` suffix so both can coexist under one output_root.
    """
    if version_prefix == "v1":
        if task == "user_interruption":
            return "user_interruption_v1"
        return task
    return task  # v1.5 tasks use bare names
