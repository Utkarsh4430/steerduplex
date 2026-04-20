"""Subprocess orchestrator for Full Duplex Bench v1.0 / v1.5 scoring.

We never import FDB eval modules directly — they pull in heavy deps (UTMOSv2,
nemo, silero_vad, statsmodels) that live in a dedicated conda env. This module
shells out to `{fdb_python} evaluate.py`, `get_timing.py`, and
`significance_test.py` with the right `cwd` + env vars, then collects the log
files FDB writes to cwd into `{output_root}/logs/`.

The scoring scripts are vendored under `fdb_v1/scoring/` so the pipeline runs
without needing the upstream `Full-Duplex-Bench/` checkout at runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.fdb_v1.dataset_utils import (
    TASKS_V1,
    TASKS_V15,
    discover_sample_dirs,
    mirror_subdir_name,
)

logger = logging.getLogger(__name__)

# Vendored upstream scoring scripts live next to this file. Subprocess `cwd`
# points here so the FDB scripts find their relative paths
# (`./instruction/behavior.txt`, `./icc_gt_distribution.json`).
DEFAULT_SCORING_DIR = Path(__file__).resolve().parent / "scoring"


# ---- Timing helpers ----------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


@contextmanager
def _stage_banner(label: str):
    bar = "=" * 72
    sys.stdout.write(f"\n{bar}\n>>> {label}\n{bar}\n")
    sys.stdout.flush()
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        sys.stdout.write(f"<<< {label} — done in {_fmt_duration(elapsed)}\n")
        sys.stdout.flush()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---- Subprocess helpers -------------------------------------------------------

def _fdb_cmd(fdb_python: str, script_cwd: Path, args: List[str]) -> List[str]:
    """Build command list for FDB subprocess.

    fdb_python may be a multi-token string (e.g. 'conda run -n full-duplex-bench python').
    """
    tokens = shlex.split(fdb_python) + args
    return tokens


def _run(fdb_python: str, cwd: Path, args: List[str], logs_dir: Path, log_name: str,
         stage_label: Optional[str] = None,
         extra_env: Optional[Dict[str, str]] = None) -> int:
    """Run an FDB subprocess with live-streamed output + log file capture.

    Output is tee'd: each line goes both to the terminal (prefixed with
    `[<log_name>]`) and to `logs_dir/log_name`. This preserves visibility
    into FDB's own tqdm progress bars and per-sample prints while still
    producing a persistent log.
    """
    cmd = _fdb_cmd(fdb_python, cwd, args)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    log_path = logs_dir / log_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    label = stage_label or log_name.replace(".log", "")
    cmd_str = " ".join(shlex.quote(t) for t in cmd)
    print(f"\n>>> {label}")
    print(f"    cmd: {cmd_str}")
    print(f"    cwd: {cwd}")
    print(f"    log: {log_path}")
    sys.stdout.flush()

    start = time.monotonic()
    tag = f"[{label}] "

    with log_path.open("w") as log_file:
        log_file.write(f"# command: {cmd_str}\n")
        log_file.write(f"# cwd:     {cwd}\n")
        log_file.write(f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
        )
        assert proc.stdout is not None

        try:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                # Forward to terminal with a short tag so interleaved stages
                # (when run sequentially) stay readable.
                sys.stdout.write(tag + line if not line.startswith(tag) else line)
                sys.stdout.flush()
        except KeyboardInterrupt:
            proc.terminate()
            raise

        returncode = proc.wait()
        elapsed = time.monotonic() - start
        log_file.write(f"\n# finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"# exit:     {returncode}\n")
        log_file.write(f"# elapsed:  {_fmt_duration(elapsed)}\n")

    status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
    print(f"<<< {label} — {status} in {_fmt_duration(elapsed)}")
    sys.stdout.flush()
    return returncode


def _move_cwd_logs_into_logs_dir(eval_cwd: Path, logs_dir: Path,
                                 pattern_globs: List[str]) -> None:
    """FDB scripts write {dirname}_{task}.log / pair_t_{dirname}.txt to cwd.
    Move any matches into our logs dir to keep eval_cwd clean between runs."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    for glob in pattern_globs:
        for path in eval_cwd.glob(glob):
            dest = logs_dir / path.name
            try:
                shutil.move(str(path), str(dest))
                logger.info("Captured FDB log: %s -> %s", path.name, dest)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to move %s: %s", path, exc)


# ---- v1.0 eval ---------------------------------------------------------------

# pause_handling unions candor + synthetic in the mirror tree, but the paper
# table reports them as separate columns. We split the union into per-source
# symlink dirs at eval time and run scoring/evaluate.py once per subset.
_V1_PAUSE_SUBSETS = ("synthetic", "candor")

_V1_LOG_NAME = {
    "pause_handling_synthetic": "v1_pause_handling_synthetic.log",
    "pause_handling_candor": "v1_pause_handling_candor.log",
    "backchannel": "v1_backchannel.log",
    "smooth_turn_taking": "v1_smooth_turn_taking.log",
    "user_interruption": "v1_user_interruption.log",
}


def _build_pause_handling_subset_views(output_root: Path) -> Dict[str, Path]:
    """Materialize per-source symlink views of the pause_handling mirror.

    Returns {subset_name: dir} for each subset that has at least one matching
    sample. The original `output_root/pause_handling/` mirror is left intact
    (inference / ASR outputs already live there); we just create
    `pause_handling_<subset>/` dirs whose entries are symlinks to the
    matching `<subset>_<id>` subdirs of the union.
    """
    base_dir = output_root / "pause_handling"
    if not base_dir.is_dir():
        return {}

    out: Dict[str, Path] = {}
    for subset in _V1_PAUSE_SUBSETS:
        subset_dir = output_root / f"pause_handling_{subset}"
        subset_dir.mkdir(exist_ok=True)
        prefix = f"{subset}_"
        n_links = 0
        for sample in sorted(base_dir.iterdir()):
            if not sample.is_dir() or not sample.name.startswith(prefix):
                continue
            link_name = sample.name[len(prefix):] or sample.name
            link = subset_dir / link_name
            if not link.exists() and not link.is_symlink():
                # Use absolute target so the view works no matter where it
                # gets read from.
                os.symlink(sample.resolve(), link)
            n_links += 1
        if n_links == 0:
            # No samples for this subset — drop the empty dir so downstream
            # discovery doesn't pick it up.
            try:
                subset_dir.rmdir()
            except OSError:
                pass
        else:
            out[subset] = subset_dir
    return out


def _run_v1_subset_eval(fdb_python: str, scoring_dir: Path, root_dir: Path,
                        log_name: str, stage_label: str, logs_dir: Path,
                        task_arg: str = "pause_handling",
                        extra_env: Optional[Dict[str, str]] = None) -> int:
    n = len(discover_sample_dirs(root_dir))
    return _run(
        fdb_python=fdb_python,
        cwd=scoring_dir,
        args=["evaluate.py", "--task", task_arg, "--root_dir", str(root_dir)],
        logs_dir=logs_dir,
        log_name=log_name,
        stage_label=f"{stage_label} ({n} samples)",
        extra_env=extra_env,
    )


def run_v1_task(fdb_python: str, scoring_dir: Path, output_root: Path, task: str,
                logs_dir: Path, extra_env: Optional[Dict[str, str]] = None) -> int:
    """Run a single non-split v1 task (backchannel / smooth_turn_taking / user_interruption)."""
    subdir = mirror_subdir_name("v1", task)
    root_dir = output_root / subdir
    if not root_dir.is_dir():
        print(f"    skip v1/{task} — {root_dir} missing")
        return 0
    return _run_v1_subset_eval(
        fdb_python=fdb_python, scoring_dir=scoring_dir, root_dir=root_dir,
        log_name=_V1_LOG_NAME[task],
        stage_label=f"v1/{task}",
        logs_dir=logs_dir,
        task_arg=task,
        extra_env=extra_env,
    )


def run_v1_all(fdb_python: str, scoring_dir: Path, output_root: Path, logs_dir: Path,
               tasks_filter: Optional[List[str]],
               litellm_base_url: Optional[str], litellm_api_key: Optional[str]) -> Dict[str, int]:
    results: Dict[str, int] = {}
    extra_env = {}
    if litellm_base_url:
        extra_env["LITELLM_BASE_URL"] = litellm_base_url
    if litellm_api_key:
        extra_env["LITELLM_API_KEY"] = litellm_api_key

    # Expand `pause_handling` (paper unioned name) into the two paper-table
    # subset evals so the user can keep writing TASKS="pause_handling".
    requested = list(tasks_filter) if tasks_filter else None
    if requested and "pause_handling" in requested:
        requested.remove("pause_handling")
        for s in _V1_PAUSE_SUBSETS:
            requested.append(f"pause_handling_{s}")

    # Single-source tasks (run as-is)
    single_tasks = [
        t for t in ("backchannel", "smooth_turn_taking", "user_interruption")
        if not requested or t in requested
    ]
    pause_subsets = [
        s for s in _V1_PAUSE_SUBSETS
        if not requested or f"pause_handling_{s}" in requested
    ]

    banner_tasks = [f"pause_handling/{s}" for s in pause_subsets] + single_tasks
    with _stage_banner(f"FDB v1.0 evaluation ({len(banner_tasks)} evals: {', '.join(banner_tasks)})"):
        if pause_subsets:
            views = _build_pause_handling_subset_views(output_root)
            for subset in pause_subsets:
                view_dir = views.get(subset)
                if view_dir is None:
                    print(f"    skip v1/pause_handling_{subset} — no samples in mirror")
                    results[f"pause_handling_{subset}"] = 0
                    continue
                results[f"pause_handling_{subset}"] = _run_v1_subset_eval(
                    fdb_python=fdb_python, scoring_dir=scoring_dir,
                    root_dir=view_dir,
                    log_name=_V1_LOG_NAME[f"pause_handling_{subset}"],
                    stage_label=f"v1/pause_handling/{subset}",
                    logs_dir=logs_dir,
                    task_arg="pause_handling",
                )
        for task in single_tasks:
            results[task] = run_v1_task(
                fdb_python=fdb_python, scoring_dir=scoring_dir, output_root=output_root,
                task=task, logs_dir=logs_dir,
                extra_env=extra_env if task == "user_interruption" else None,
            )
    return results


# ---- v1.5 eval ---------------------------------------------------------------

def run_v15_task(fdb_python: str, scoring_dir: Path, output_root: Path, task: str,
                 logs_dir: Path,
                 litellm_base_url: Optional[str], litellm_api_key: Optional[str]) -> Dict[str, int]:
    eval_cwd = scoring_dir
    subdir = mirror_subdir_name("v15", task)
    root_dir = output_root / subdir
    if not root_dir.is_dir():
        print(f"    skip v1.5/{task} — {root_dir} missing")
        return {"behavior": 0, "timing": 0, "general": 0, "significance": 0}

    extra_env = {}
    if litellm_base_url:
        extra_env["LITELLM_BASE_URL"] = litellm_base_url
    if litellm_api_key:
        extra_env["LITELLM_API_KEY"] = litellm_api_key

    n = len(discover_sample_dirs(root_dir))
    results: Dict[str, int] = {}

    with _stage_banner(f"v1.5/{task} ({n} samples) — 4 stages (behavior, timing, general, significance)"):
        # Order matters: behavior writes content_tag.json, general/significance use it.
        results["behavior"] = _run(
            fdb_python=fdb_python, cwd=eval_cwd,
            args=["evaluate.py", "--task", "behavior", "--root_dir", str(root_dir)],
            logs_dir=logs_dir, log_name=f"v15_{task}_behavior.log",
            stage_label=f"v1.5/{task}/behavior",
            extra_env=extra_env,
        )
        _move_cwd_logs_into_logs_dir(eval_cwd, logs_dir, [f"{root_dir.name}_behavior.log"])

        results["timing"] = _run(
            fdb_python=fdb_python, cwd=eval_cwd,
            args=["get_timing.py", "--root_dir", str(root_dir)],
            logs_dir=logs_dir, log_name=f"v15_{task}_timing.log",
            stage_label=f"v1.5/{task}/timing",
        )

        results["general"] = _run(
            fdb_python=fdb_python, cwd=eval_cwd,
            args=["evaluate.py", "--task", "general_before_after", "--root_dir", str(root_dir)],
            logs_dir=logs_dir, log_name=f"v15_{task}_general.log",
            stage_label=f"v1.5/{task}/general_before_after",
        )
        _move_cwd_logs_into_logs_dir(eval_cwd, logs_dir, [f"{root_dir.name}_general.log"])

        results["significance"] = _run(
            fdb_python=fdb_python, cwd=eval_cwd,
            args=["significance_test.py", "--root_dir", str(root_dir),
                  "--metrics", "utmosv2", "wpm", "mean_pitch", "std_pitch",
                  "mean_intensity", "std_intensity",
                  "--outlier_rule", "mad"],
            logs_dir=logs_dir, log_name=f"v15_{task}_significance.log",
            stage_label=f"v1.5/{task}/significance",
        )
        _move_cwd_logs_into_logs_dir(eval_cwd, logs_dir, [f"pair_t_{root_dir.name}*.txt"])

    return results


def run_v15_all(fdb_python: str, scoring_dir: Path, output_root: Path, logs_dir: Path,
                tasks_filter: Optional[List[str]],
                litellm_base_url: Optional[str], litellm_api_key: Optional[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    tasks = [t for t in TASKS_V15.keys() if not tasks_filter or t in tasks_filter]
    with _stage_banner(f"FDB v1.5 evaluation ({len(tasks)} tasks × 4 stages: {', '.join(tasks)})"):
        for task in tasks:
            out[task] = run_v15_task(
                fdb_python=fdb_python, scoring_dir=scoring_dir, output_root=output_root,
                task=task, logs_dir=logs_dir,
                litellm_base_url=litellm_base_url, litellm_api_key=litellm_api_key,
            )
    return out


# ---- Summary aggregation -----------------------------------------------------

def _parse_log_scalars(log_path: Path) -> Dict[str, float]:
    """Extract metric scalars from FDB v1 eval logs.

    Handles both formats the upstream scripts emit:
      - "Average <metric>: <number>"            (pause_handling, smooth_turn_taking, user_interruption)
      - "<metric> - Mean: <number> ± <stddev>"  (backchannel)

    Keys are lowercased + spaces collapsed. Backchannel metrics map to
    'tor', 'jsd', 'frequency'; the others to 'take_turn', 'latency', 'rating'.
    """
    if not log_path.exists():
        return {}
    out: Dict[str, float] = {}
    avg_pattern = re.compile(r"Average\s+([^:]+):\s*([\-0-9eE+.]+)")
    mean_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _]*?)\s*-\s*Mean:\s*([\-0-9eE+.]+)")
    for line in log_path.read_text().splitlines():
        m = avg_pattern.search(line)
        if m:
            key = m.group(1).strip().replace(" ", "_").lower()
            try:
                out[key] = float(m.group(2))
            except ValueError:
                pass
            continue
        m = mean_pattern.search(line)
        if m:
            key = m.group(1).strip().replace(" ", "_").lower()
            try:
                out[key] = float(m.group(2))
            except ValueError:
                pass
    return out


def _parse_behavior_log(log_path: Path) -> Dict[str, float]:
    """Parse FDB behavior log: `Ratios (C-axis): {'C_RESPOND': 0.5, ...}`."""
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    m = re.search(r"Ratios \([CV]-axis\):\s*(\{.*?\})", text, re.DOTALL)
    if not m:
        return {}
    try:
        # Python dict literal → replace single quotes to parse as JSON
        blob = m.group(1).replace("'", '"')
        return json.loads(blob)
    except Exception:
        return {}


def _parse_general_log(log_path: Path) -> Dict[str, float]:
    """Parse `key: value` lines from general_before_after logs."""
    if not log_path.exists():
        return {}
    out: Dict[str, float] = {}
    for line in log_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        try:
            out[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return out


def _collect_sample_counts(output_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    # Pause handling has two paper-table subset views (synthetic / candor)
    # built at eval time; report both alongside the union.
    for subset in _V1_PAUSE_SUBSETS:
        d = output_root / f"pause_handling_{subset}"
        counts[f"v1/pause_handling_{subset}"] = len(discover_sample_dirs(d)) if d.is_dir() else 0
    for task in ("backchannel", "smooth_turn_taking", "user_interruption"):
        d = output_root / mirror_subdir_name("v1", task)
        counts[f"v1/{task}"] = len(discover_sample_dirs(d)) if d.is_dir() else 0
    for task in TASKS_V15.keys():
        d = output_root / mirror_subdir_name("v15", task)
        counts[f"v15/{task}"] = len(discover_sample_dirs(d)) if d.is_dir() else 0
    return counts


def build_summary(output_root: Path, logs_dir: Path) -> Dict:
    summary: Dict = {"v1": {}, "v15": {}, "sample_counts": _collect_sample_counts(output_root)}

    # v1 metrics: parse one log per (subset-aware) task name.
    for log_key, log_name in _V1_LOG_NAME.items():
        summary["v1"][log_key] = _parse_log_scalars(logs_dir / log_name)

    for task in TASKS_V15.keys():
        task_subdir = mirror_subdir_name("v15", task)
        behavior = _parse_behavior_log(logs_dir / f"{task_subdir}_behavior.log")
        general = _parse_general_log(logs_dir / f"{task_subdir}_general.log")
        summary["v15"][task] = {"behavior": behavior, "general": general}

    return summary


# ---- Paper-format v1 table ---------------------------------------------------
#
# Mirrors the published Full-Duplex-Bench v1.0 headline table. Each cell is
# (Dimension, Data, Metric, Direction, summary_log_key, scalar_key_in_log).
# Direction determines what is printed in the column header (↑ / ↓) and is
# also used by `_select_best_per_column` to bold the leading model.

_V1_TABLE_CELLS = [
    # Dimension              Data         Metric      Dir   log key                      scalar
    ("Pause Handling",      "Synthetic", "TOR",      "↓",  "pause_handling_synthetic",  "take_turn"),
    ("Pause Handling",      "Candor",    "TOR",      "↓",  "pause_handling_candor",     "take_turn"),
    ("Backchannel",         "ICC",       "TOR",      "↓",  "backchannel",               "tor"),
    ("Backchannel",         "ICC",       "Freq",     "↑",  "backchannel",               "frequency"),
    ("Backchannel",         "ICC",       "JSD",      "↓",  "backchannel",               "jsd"),
    ("Smooth Turn Taking",  "Candor",    "TOR",      "↑",  "smooth_turn_taking",        "take_turn"),
    ("Smooth Turn Taking",  "Candor",    "Latency",  "↓",  "smooth_turn_taking",        "latency"),
    ("User Interruption",   "Synthetic", "TOR",      "↑",  "user_interruption",         "take_turn"),
    ("User Interruption",   "Synthetic", "GPT-4o",   "↑",  "user_interruption",         "rating"),
    ("User Interruption",   "Synthetic", "Latency",  "↓",  "user_interruption",         "latency"),
]


def _row_values_v1(summary_v1: Dict[str, Dict[str, float]]) -> List[Optional[float]]:
    """Pluck the v1 table cells out of build_summary['v1'] in column order."""
    out: List[Optional[float]] = []
    for _, _, _, _, log_key, scalar in _V1_TABLE_CELLS:
        scalars = summary_v1.get(log_key, {}) or {}
        v = scalars.get(scalar)
        out.append(v if isinstance(v, (int, float)) else None)
    return out


def _model_label(output_root: Path) -> str:
    """Use the model_name recorded by run_inference.py if present."""
    meta = output_root / "run_meta.json"
    if meta.is_file():
        try:
            data = json.loads(meta.read_text())
            return str(data.get("model_name") or output_root.name)
        except Exception:
            pass
    return output_root.name


def _render_v1_paper_table(model_label: str, values: List[Optional[float]]) -> str:
    """Emit the paper's two-row-grouped headline table as HTML.

    Markdown can't do colspan / multi-row headers, so we hand-write a small
    HTML <table>; GitHub markdown, VS Code preview, and most renderers render
    it cleanly inline.
    """
    # Group cells by Dimension preserving order
    dims: List[Tuple[str, List[int]]] = []  # (dimension, [col_indices])
    for i, (dim, *_rest) in enumerate(_V1_TABLE_CELLS):
        if dims and dims[-1][0] == dim:
            dims[-1][1].append(i)
        else:
            dims.append((dim, [i]))
    # Group cells within each dimension by Data
    data_groups: List[List[Tuple[str, List[int]]]] = []
    for _, idxs in dims:
        groups: List[Tuple[str, List[int]]] = []
        for i in idxs:
            data = _V1_TABLE_CELLS[i][1]
            if groups and groups[-1][0] == data:
                groups[-1][1].append(i)
            else:
                groups.append((data, [i]))
        data_groups.append(groups)

    def fmt(v: Optional[float], scale: int = 3) -> str:
        if v is None:
            return "—"
        return f"{v:.{scale}f}"

    parts: List[str] = []
    parts.append("<table>")
    parts.append("  <thead>")

    # Row 1: Dimension headers (colspan = number of metric columns under it)
    parts.append("    <tr>")
    parts.append('      <th rowspan="3">Model</th>')
    for dim, idxs in dims:
        parts.append(f'      <th colspan="{len(idxs)}">{dim}</th>')
    parts.append("    </tr>")

    # Row 2: Data headers (colspan = number of metric columns in this data group)
    parts.append("    <tr>")
    for groups in data_groups:
        for data, idxs in groups:
            parts.append(f'      <th colspan="{len(idxs)}">{data}</th>')
    parts.append("    </tr>")

    # Row 3: Metric (Direction)
    parts.append("    <tr>")
    for _, _, metric, direction, _lk, _sk in _V1_TABLE_CELLS:
        parts.append(f"      <th>{metric} ({direction})</th>")
    parts.append("    </tr>")

    parts.append("  </thead>")
    parts.append("  <tbody>")
    parts.append("    <tr>")
    parts.append(f"      <td>{model_label}</td>")
    for v in values:
        parts.append(f"      <td>{fmt(v)}</td>")
    parts.append("    </tr>")
    parts.append("  </tbody>")
    parts.append("</table>")
    return "\n".join(parts)


def write_summary(output_root: Path, logs_dir: Path) -> None:
    summary = build_summary(output_root, logs_dir)

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))

    model_label = _model_label(output_root)
    lines: List[str] = [f"# Full Duplex Bench — {model_label}", ""]

    # ---- v1.0 paper-format headline table ------------------------------------
    if any(summary["v1"].get(k) for k in (
        "pause_handling_synthetic", "pause_handling_candor",
        "backchannel", "smooth_turn_taking", "user_interruption",
    )):
        lines.append("## v1.0")
        lines.append("")
        lines.append(_render_v1_paper_table(model_label, _row_values_v1(summary["v1"])))
        lines.append("")

    # ---- v1.5 (placeholder until v1.5 paper table is wired up) ---------------
    if summary["v15"]:
        lines.append("## v1.5")
        lines.append("")
        for task, payload in summary["v15"].items():
            lines.append(f"### {task}")
            lines.append("")
            if payload.get("behavior"):
                lines.append("**Behavior (C-axis ratios):**")
                for k, v in payload["behavior"].items():
                    lines.append(f"- {k}: {v:.3f}")
            else:
                lines.append("_no behavior log_")
            lines.append("")
            if payload.get("general"):
                lines.append("**General before/after:**")
                for k, v in payload["general"].items():
                    lines.append(f"- {k}: {v}")
            else:
                lines.append("_no general log_")
            lines.append("")

    # ---- Sample counts (collapsed below for reference) -----------------------
    lines.append("## Sample counts")
    lines.append("")
    lines.append("| Task | Samples |")
    lines.append("|------|---------|")
    for k, v in summary["sample_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    (output_root / "summary.md").write_text("\n".join(lines))
    logger.info("Wrote summary.json and summary.md to %s", output_root)


# ---- CLI ---------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FDB v1/v1.5 evaluation orchestrator")
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--scoring_dir", default=str(DEFAULT_SCORING_DIR),
                        help=f"Path to vendored FDB scoring scripts (default: {DEFAULT_SCORING_DIR})")
    parser.add_argument("--fdb_python", default="python",
                        help="Command to invoke FDB env's python (e.g. 'conda run -n raman_fdb_v1 python')")
    parser.add_argument("--version", default="both", choices=["1.0", "1.5", "both"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--litellm_base_url", default=os.environ.get("LITELLM_BASE_URL"))
    parser.add_argument("--litellm_api_key", default=os.environ.get("LITELLM_API_KEY"))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    setup_logging()
    args = parse_args(argv)

    output_root = Path(args.output_root).resolve()
    scoring_dir = Path(args.scoring_dir).resolve()
    if not scoring_dir.is_dir():
        raise SystemExit(f"scoring_dir not found: {scoring_dir}")
    if not (scoring_dir / "evaluate.py").is_file():
        raise SystemExit(f"scoring_dir missing evaluate.py: {scoring_dir}")
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.monotonic()
    print(f"FDB evaluation orchestrator")
    print(f"  output_root: {output_root}")
    print(f"  scoring_dir: {scoring_dir}")
    print(f"  fdb_python:  {args.fdb_python}")
    print(f"  version:     {args.version}")
    print(f"  tasks:       {args.tasks or 'all'}")
    print(f"  logs dir:    {logs_dir}")

    if args.version in ("1.0", "both"):
        run_v1_all(
            fdb_python=args.fdb_python, scoring_dir=scoring_dir, output_root=output_root,
            logs_dir=logs_dir, tasks_filter=args.tasks,
            litellm_base_url=args.litellm_base_url,
            litellm_api_key=args.litellm_api_key,
        )

    if args.version in ("1.5", "both"):
        run_v15_all(
            fdb_python=args.fdb_python, scoring_dir=scoring_dir, output_root=output_root,
            logs_dir=logs_dir, tasks_filter=args.tasks,
            litellm_base_url=args.litellm_base_url,
            litellm_api_key=args.litellm_api_key,
        )

    write_summary(output_root, logs_dir)
    total = time.monotonic() - overall_start
    print(f"\nFDB evaluation complete in {_fmt_duration(total)}.")
    print(f"  summary:  {output_root / 'summary.md'}")
    print(f"  logs:     {logs_dir}")


if __name__ == "__main__":
    main()
