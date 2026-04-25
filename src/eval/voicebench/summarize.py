"""Phase 3: aggregate judged results into a `summary.md` table.

Delegates scoring to VoiceBench's official evaluator classes (from
/mnt/efs/ramaneswaranselvakumar/projects/steerduplex/VoiceBench/src/evaluator)
by adding that path to sys.path — avoids vendoring, keeps leaderboard parity.

The canonical `response` field given to VoiceBench scorers is the Whisper
transcription of the Moshi output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.voicebench.splits import (
    SDQA_DEFAULT_REGIONS,
    SUPPORTED_SUBSETS,
    SplitSpec,
    build_specs,
)

VOICEBENCH_SRC = Path("/mnt/efs/ramaneswaranselvakumar/projects/steerduplex/VoiceBench/src")

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _import_voicebench_evaluators():
    """Return VoiceBench's `evaluator_mapping` dict, importing lazily.

    Adds VoiceBench/src to sys.path once. Expects `qa_metrics` and `loguru`
    to be available in the active environment.
    """
    if not VOICEBENCH_SRC.exists():
        raise FileNotFoundError(
            f"VoiceBench/src not found at {VOICEBENCH_SRC}. "
            "Clone https://github.com/MatthewCYM/VoiceBench next to this repo."
        )
    if str(VOICEBENCH_SRC) not in sys.path:
        sys.path.insert(0, str(VOICEBENCH_SRC))
    from evaluator import evaluator_mapping  # type: ignore
    return evaluator_mapping


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def _prepare_for_evaluator(spec: SplitSpec, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rename/copy fields so VoiceBench's evaluator classes see what they expect.

    VoiceBench reads `response` (plain str). Our records carry `transcription`.
    Open/QA records without a judge score are filtered out — the VoiceBench
    evaluators iterate `item['score']` unconditionally and one missing field
    would KeyError the whole split.
    """
    out: List[Dict[str, Any]] = []
    dropped_no_transcription = 0
    dropped_no_score = 0
    for r in records:
        if r.get("transcription") is None:
            dropped_no_transcription += 1
            continue
        if spec.evaluator in ("open", "qa") and not r.get("score"):
            dropped_no_score += 1
            continue
        item: Dict[str, Any] = {
            "prompt": r.get("prompt", ""),
            "response": r.get("transcription", ""),
        }
        if spec.evaluator == "open":
            item["score"] = r["score"]
        if spec.evaluator == "qa":
            item["reference"] = r.get("reference", "")
            item["score"] = r["score"]
        if spec.evaluator == "bbh":
            item["reference"] = r.get("reference", "")
            item["id"] = r.get("id", "")
        if spec.evaluator == "ifeval":
            item["key"] = r.get("key")
            item["instruction_id_list"] = r.get("instruction_id_list", [])
            item["kwargs"] = r.get("kwargs", [])
        out.append(item)
    if dropped_no_transcription or dropped_no_score:
        logger.info(
            "[%s] %d scorable / dropped: %d no-transcription, %d no-score.",
            spec.name, len(out), dropped_no_transcription, dropped_no_score,
        )
    return out


def _score_spec(spec: SplitSpec, evaluator_mapping: Dict[str, Any],
                records: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    data = _prepare_for_evaluator(spec, records)
    if not data:
        logger.warning("[%s] No scorable records. Skipping.", spec.name)
        return None
    try:
        evaluator = evaluator_mapping[spec.evaluator]()
        return evaluator.evaluate(data)
    except Exception:
        logger.exception("[%s] Scorer failed.", spec.name)
        return None


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

# Describes how each evaluator's dict maps to rows in the per-split details table.
_METRIC_RENDER = {
    "open":   [("gpt",               "GPT (1–5)",     lambda v: f"{v:.2f}")],
    "qa":     [("gpt",               "GPT yes %",     lambda v: f"{v:.1f}"),
               ("panda",             "PEDANT %",      lambda v: f"{v:.1f}")],
    "ifeval": [("strict-prompt",     "strict-prompt %",     lambda v: f"{v*100:.1f}"),
               ("strict-instruction","strict-instruction %", lambda v: f"{v*100:.1f}"),
               ("loose-prompt",      "loose-prompt %",       lambda v: f"{v*100:.1f}"),
               ("loose-instruction", "loose-instruction %",  lambda v: f"{v*100:.1f}"),
               ("final",             "final (mean)",         lambda v: f"{v*100:.1f}")],
    "bbh":    [("acc",               "accuracy %",    lambda v: f"{v:.1f}")],
    "harm":   [("refusal_rate",      "refusal rate %",lambda v: f"{v*100:.1f}")],
}


# Order and definition of the paper-style leaderboard columns.
# Each entry: (column header, subset key in per_spec_metrics, raw→display, raw→normalized[0,100]).
# "subset_key" is matched against `spec.subset`; sd-qa rows are averaged across regions if >1.
# MMSU and OpenBookQA are deliberately missing here — this harness skips MCQ splits.
_PAPER_COLUMNS: List[Tuple[str, str]] = [
    ("AlpacaEval",    "alpacaeval_full"),
    ("CommonEval",    "commoneval"),
    ("WildVoice",     "wildvoice"),
    ("SD-QA (G/P)",   "sd-qa"),
    ("MMSU",          "__mmsu__"),        # sentinel — never populated
    ("OpenBookQA",    "__openbookqa__"),  # sentinel — never populated
    ("IFEval (P/I)",  "ifeval"),
    ("BBH",           "bbh"),
    ("AdvBench",      "advbench"),
]


def _collect_by_subset(
    specs: List[SplitSpec],
    per_spec_metrics: Dict[str, Optional[Dict[str, float]]],
) -> Dict[str, List[Dict[str, float]]]:
    """Group non-empty metric dicts by `spec.subset` (so sd-qa regions collapse together)."""
    by_subset: Dict[str, List[Dict[str, float]]] = {}
    for spec in specs:
        metrics = per_spec_metrics.get(spec.name)
        if not metrics:
            continue
        by_subset.setdefault(spec.subset, []).append(metrics)
    return by_subset


def _build_paper_row(
    by_subset: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Dict[str, Any]]:
    """For each paper column, return {"display": str, "normalized": float or None}.

    Normalization targets [0, 100] so all columns average on the same scale.
    """
    row: Dict[str, Dict[str, Any]] = {}
    for header, subset_key in _PAPER_COLUMNS:
        runs = by_subset.get(subset_key, [])
        if not runs:
            row[header] = {"display": "—", "normalized": None}
            continue

        if subset_key in ("alpacaeval_full", "commoneval", "wildvoice"):
            gpts = [m["gpt"] for m in runs if "gpt" in m]
            if not gpts:
                row[header] = {"display": "—", "normalized": None}
                continue
            val = sum(gpts) / len(gpts)
            row[header] = {"display": f"{val:.2f}", "normalized": val * 20.0}

        elif subset_key == "sd-qa":
            gpts  = [m["gpt"]   for m in runs if "gpt"   in m]
            pands = [m["panda"] for m in runs if "panda" in m]
            if not gpts or not pands:
                row[header] = {"display": "—", "normalized": None}
                continue
            g = sum(gpts)  / len(gpts)
            p = sum(pands) / len(pands)
            row[header] = {"display": f"{g:.1f}/{p:.1f}", "normalized": g}

        elif subset_key == "ifeval":
            m = runs[0]
            sp  = m.get("strict-prompt")
            si  = m.get("strict-instruction")
            fin = m.get("final")
            if sp is None or si is None or fin is None:
                row[header] = {"display": "—", "normalized": None}
                continue
            row[header] = {
                "display":    f"{sp*100:.1f}/{si*100:.1f}",
                "normalized": fin * 100.0,
            }

        elif subset_key == "bbh":
            m = runs[0]
            if "acc" not in m:
                row[header] = {"display": "—", "normalized": None}
                continue
            row[header] = {"display": f"{m['acc']:.1f}", "normalized": m["acc"]}

        elif subset_key == "advbench":
            m = runs[0]
            if "refusal_rate" not in m:
                row[header] = {"display": "—", "normalized": None}
                continue
            r = m["refusal_rate"] * 100.0
            row[header] = {"display": f"{r:.1f}", "normalized": r}

        else:  # MMSU / OpenBookQA sentinels — not supported by this harness.
            row[header] = {"display": "—", "normalized": None}

    return row


def _compute_overall(row: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """Mean of the normalized column values that are present (divisor = count available).

    The VoiceBench paper divides by 9; we divide by the number of columns actually
    run so a partial split set still gives a comparable number. Footnoted in the
    rendered markdown to make the difference clear.
    """
    vals = [c["normalized"] for c in row.values() if c["normalized"] is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _render_markdown(
    output_dir: Path,
    specs: List[SplitSpec],
    per_spec_metrics: Dict[str, Optional[Dict[str, float]]],
    per_spec_counts: Dict[str, int],
    run_args: Dict[str, Any],
    paper_row: Dict[str, Dict[str, Any]],
    overall: Optional[float],
) -> str:
    lines: List[str] = []
    lines.append("# VoiceBench Results\n")
    lines.append(f"- Date: {datetime.now().isoformat(timespec='seconds')}")
    model = run_args.get("hf_repo", "?")
    if run_args.get("checkpoint"):
        model = f"{model} @ {run_args['checkpoint']}"
    lines.append(f"- Model: {model}")
    lines.append(f"- System prompt: {run_args.get('system_prompt', '')!r}")
    lines.append(
        f"- Config: instances_per_gpu={run_args.get('instances_per_gpu')}, "
        f"max_duration={run_args.get('max_duration')}, greedy={run_args.get('greedy')}, "
        f"seed={run_args.get('seed')}"
    )
    lines.append("")

    # Paper-style leaderboard table.
    headers = [h for h, _ in _PAPER_COLUMNS]
    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| Model | " + " | ".join(headers) + " | Overall |")
    lines.append("|---" + "|---:" * (len(headers) + 1) + "|")
    cells = [paper_row[h]["display"] for h in headers]
    overall_cell = f"{overall:.2f}" if overall is not None else "—"
    lines.append(f"| {model} | " + " | ".join(cells) + f" | {overall_cell} |")
    lines.append("")

    missing = [h for h in headers if paper_row[h]["normalized"] is None]
    present = [h for h in headers if paper_row[h]["normalized"] is not None]
    note_bits = [
        "Overall = arithmetic mean of normalized column values "
        "(1–5 scores ×20; IFEval final ×100; AdvBench refusal ×100; others as-is).",
        f"Denominator = {len(present)} (columns with data); "
        "this harness does not evaluate MMSU/OpenBookQA, so Overall is not comparable "
        "to the paper's 9-split Overall when those columns are missing.",
    ]
    if missing:
        note_bits.append("Missing columns: " + ", ".join(missing) + ".")
    lines.append("> " + " ".join(note_bits))
    lines.append("")

    # Per-split details (stacked view — preserves every sub-metric).
    lines.append("## Per-split details")
    lines.append("")
    lines.append("| Split | N | Metric | Score |")
    lines.append("|---|---:|---|---:|")
    for spec in specs:
        metrics = per_spec_metrics.get(spec.name)
        count = per_spec_counts.get(spec.name, 0)
        if metrics is None:
            lines.append(f"| {spec.name} | {count} | — | (not scored) |")
            continue
        for key, label, fmt in _METRIC_RENDER.get(spec.evaluator, []):
            if key in metrics:
                lines.append(f"| {spec.name} | {count} | {label} | {fmt(metrics[key])} |")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def write_summary(
    output_dir: Path,
    specs: List[SplitSpec],
    run_args: Dict[str, Any],
) -> Path:
    setup_logging()
    evaluator_mapping = _import_voicebench_evaluators()

    per_spec_metrics: Dict[str, Optional[Dict[str, float]]] = {}
    per_spec_counts: Dict[str, int] = {}
    for spec in specs:
        judged_json = output_dir / spec.name / "llm_judge_output.json"
        if not judged_json.exists():
            logger.warning("[%s] %s not found, skipping.", spec.name, judged_json)
            per_spec_metrics[spec.name] = None
            per_spec_counts[spec.name] = 0
            continue
        records = load_records(judged_json)
        per_spec_counts[spec.name] = sum(1 for r in records if r.get("transcription"))
        logger.info("[%s] Scoring %d records...", spec.name, per_spec_counts[spec.name])
        per_spec_metrics[spec.name] = _score_spec(spec, evaluator_mapping, records)
        if per_spec_metrics[spec.name]:
            logger.info("[%s] → %s", spec.name, per_spec_metrics[spec.name])

    by_subset = _collect_by_subset(specs, per_spec_metrics)
    paper_row = _build_paper_row(by_subset)
    overall = _compute_overall(paper_row)

    md = _render_markdown(
        output_dir, specs, per_spec_metrics, per_spec_counts, run_args, paper_row, overall,
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text(md)
    logger.info("Wrote %s", summary_path)

    # Also dump metrics to JSON for programmatic consumption.
    (output_dir / "summary.json").write_text(json.dumps({
        "per_split": {spec.name: per_spec_metrics.get(spec.name) for spec in specs},
        "paper_row": {h: paper_row[h] for h, _ in _PAPER_COLUMNS},
        "overall": overall,
    }, indent=2))
    return summary_path


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceBench Phase 3 (summary.md)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=list(SUPPORTED_SUBSETS))
    parser.add_argument("--sdqa_regions", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    output_dir = Path(args.output_dir)

    sdqa_regions = args.sdqa_regions
    if "sd-qa" in args.splits and sdqa_regions is None:
        sdqa_regions = list(SDQA_DEFAULT_REGIONS)
    specs = build_specs(args.splits, sdqa_regions or [])

    # Try to recover run_args from run_manifest.json for richer header.
    manifest_path = output_dir / "run_manifest.json"
    run_args: Dict[str, Any] = {}
    if manifest_path.exists():
        run_args = json.loads(manifest_path.read_text())

    write_summary(output_dir, specs, run_args)


if __name__ == "__main__":
    main()
