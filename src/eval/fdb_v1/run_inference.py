"""Server-based Moshi inference for Full Duplex Bench v1.0 / v1.5.

Talks to one or more Moshi WebSocket servers (`moshi_server.py`, launched
via `launch_moshi_servers.py`). Each server is a single-stream channel; we
distribute samples across servers with a small asyncio worker pool so multiple
samples can be in-flight concurrently when more than one server is up.

Server config (model, checkpoint, cfg-coef, sampling) lives entirely on the
server side. This client only carries audio over the wire and writes outputs
in-place next to the symlinked dataset inputs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from eval.fdb_v1.dataset_utils import (
    TASKS_V1,
    TASKS_V15,
    build_mirror_for_task,
    inference_done_for_sample,
    mirror_subdir_name,
)
from eval.fdb_v1.moshi_client import MoshiFileClient

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---- Per-sample inference -----------------------------------------------------

async def _generate_one(ws_url: str, input_wav: Path, output_wav: Path) -> None:
    tmp = output_wav.with_suffix(".wav.tmp")
    try:
        await MoshiFileClient(ws_url, input_wav, tmp).run()
        os.replace(tmp, output_wav)
    finally:
        # tmp may exist if the run died mid-stream; soundfile leaves a
        # partial WAV. Drop it so resume logic doesn't trip on zero-length
        # outputs.
        if tmp.exists() and not output_wav.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


async def _run_one_sample(ws_url: str, sample_dir: Path, paired: bool,
                          overwrite: bool) -> Tuple[bool, Optional[str], float]:
    t0 = time.monotonic()
    try:
        if overwrite or not (sample_dir / "output.wav").exists():
            await _generate_one(ws_url, sample_dir / "input.wav",
                                sample_dir / "output.wav")
        clean_input = sample_dir / "clean_input.wav"
        if paired and clean_input.exists():
            clean_output = sample_dir / "clean_output.wav"
            if overwrite or not clean_output.exists():
                await _generate_one(ws_url, clean_input, clean_output)
        return True, None, time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", time.monotonic() - t0


# ---- Task-level sample collection ---------------------------------------------

def _collect_tasks(version: str, tasks_filter: Optional[List[str]]) -> List[Tuple[str, str, Dict]]:
    out: List[Tuple[str, str, Dict]] = []
    if version in ("1.0", "both"):
        for task, meta in TASKS_V1.items():
            if tasks_filter and task not in tasks_filter:
                continue
            out.append(("v1", task, meta))
    if version in ("1.5", "both"):
        for task, meta in TASKS_V15.items():
            if tasks_filter and task not in tasks_filter:
                continue
            out.append(("v15", task, meta))
    return out


def _prepare_samples(dataset_root: Path, output_root: Path, version: str,
                     tasks_filter: Optional[List[str]], use_copy: bool,
                     overwrite: bool) -> List[Tuple[Path, bool, str]]:
    all_samples: List[Tuple[Path, bool, str]] = []
    task_counts: List[Tuple[str, int, int, int]] = []
    for version_prefix, task, meta in _collect_tasks(version, tasks_filter):
        subdir = mirror_subdir_name(version_prefix, task)
        mirrored = build_mirror_for_task(
            dataset_root=dataset_root,
            output_root=output_root,
            task=task,
            task_meta=meta,
            use_copy=use_copy,
            out_subdir=subdir,
            version_prefix=version_prefix,
        )
        paired = bool(meta.get("paired", False))
        pending = 0
        skipped = 0
        for sample_dir in mirrored:
            if not overwrite and inference_done_for_sample(sample_dir, paired):
                skipped += 1
                continue
            all_samples.append((sample_dir, paired, f"{version_prefix}/{task}"))
            pending += 1
        task_counts.append((f"{version_prefix}/{task}", len(mirrored), pending, skipped))

    print("")
    print(f"{'Task':30s}  {'Mirrored':>9s}  {'Pending':>8s}  {'Resumed':>8s}")
    print("-" * 62)
    for label, mirrored_n, pending_n, skipped_n in task_counts:
        print(f"  {label:28s}  {mirrored_n:>9d}  {pending_n:>8d}  {skipped_n:>8d}")
    print("-" * 62)
    print(f"  {'TOTAL':28s}  {sum(t[1] for t in task_counts):>9d}  "
          f"{sum(t[2] for t in task_counts):>8d}  "
          f"{sum(t[3] for t in task_counts):>8d}")
    sys.stdout.flush()
    return all_samples


# ---- Server load-balanced async runner ---------------------------------------

async def _worker(name: str, ws_url: str, queue: asyncio.Queue,
                  overwrite: bool, on_result) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        sample_dir, paired, task_label = item
        ok, err, elapsed = await _run_one_sample(ws_url, sample_dir, paired, overwrite)
        on_result(name, task_label, sample_dir, ok, err, elapsed)
        queue.task_done()


async def _drive(servers: List[str], samples: List[Tuple[Path, bool, str]],
                 overwrite: bool) -> Dict[str, int]:
    """Distribute `samples` across `servers` (one in-flight stream per URL)."""
    stats: Dict[str, int] = {"ok": 0, "err": 0, "total": len(samples)}
    per_task: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ok": 0, "err": 0})

    queue: asyncio.Queue = asyncio.Queue()
    for s in samples:
        queue.put_nowait(s)
    for _ in servers:
        queue.put_nowait(None)  # one sentinel per worker

    pbar = tqdm(total=len(samples), desc="Moshi FDB inference", unit="sample",
                dynamic_ncols=True)

    def on_result(worker_name: str, task_label: str, sample_dir: Path,
                  ok: bool, err: Optional[str], elapsed: float) -> None:
        if ok:
            stats["ok"] += 1
            per_task[task_label]["ok"] += 1
        else:
            stats["err"] += 1
            per_task[task_label]["err"] += 1
            tqdm.write(f"ERROR [{worker_name}] [{task_label}] {sample_dir}: {err}")
        pbar.update(1)
        pbar.set_postfix({
            "task": task_label,
            "last": f"{elapsed:.1f}s",
            "ok": stats["ok"],
            "err": stats["err"],
        })

    workers = [
        asyncio.create_task(_worker(f"srv{i}", url, queue, overwrite, on_result))
        for i, url in enumerate(servers)
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        pbar.close()

    print("")
    if per_task:
        print("Per-task breakdown:")
        for task_label in sorted(per_task):
            counts = per_task[task_label]
            print(f"  {task_label:30s}  ok={counts['ok']:4d}  err={counts['err']:4d}")
    return stats


def run_inference(servers: List[str], samples: List[Tuple[Path, bool, str]],
                  overwrite: bool) -> Dict[str, int]:
    if not samples:
        logger.info("No pending samples — everything already has outputs.")
        return {"ok": 0, "err": 0, "total": 0}
    if not servers:
        raise ValueError("At least one Moshi server URL must be provided.")

    wall_start = time.monotonic()
    stats = asyncio.run(_drive(servers, samples, overwrite))
    wall = time.monotonic() - wall_start
    print(f"Inference done in {_fmt_duration(wall)} "
          f"({stats['ok']}/{stats['total']} ok, {stats['err']} err)")
    sys.stdout.flush()
    return stats


# ---- Servers JSON helper -----------------------------------------------------

def _load_servers(servers_json: Optional[str], explicit: Optional[List[str]]) -> List[str]:
    """Resolve the list of WebSocket URLs.

    Priority: --server (explicit list) > --servers_json (file produced by
    launch_moshi_servers.py).
    """
    if explicit:
        return list(explicit)
    if servers_json:
        path = Path(servers_json).resolve()
        if not path.is_file():
            raise SystemExit(f"--servers_json not found: {path}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Could not parse {path}: {exc}") from exc
        urls = data.get("servers")
        if not urls:
            raise SystemExit(f"No 'servers' key (or empty) in {path}")
        return list(urls)
    raise SystemExit("Pass either --servers_json or --server URL [URL ...].")


# ---- CLI ---------------------------------------------------------------------

def _git_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _write_run_meta(output_root: Path, args: argparse.Namespace,
                    servers: List[str], sample_count: int, stats: Dict[str, int]) -> None:
    steerduplex_repo = Path(__file__).resolve().parents[3]  # .../steerduplex/
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "hf_repo": args.hf_repo,
        "checkpoint": args.checkpoint,
        "cfg_coef": args.cfg_coef,
        "seed": args.seed,
        "version": args.version,
        "tasks_filter": args.tasks,
        "servers": servers,
        "dataset_root": args.dataset_root,
        "steerduplex_git_sha": _git_sha(steerduplex_repo),
        "sample_count": sample_count,
        "stats": stats,
    }
    path = output_root / "run_meta.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FDB v1/v1.5 Moshi inference (server-based)")
    parser.add_argument("--dataset_root", required=True, type=str,
                        help="FDB dataset root containing {task}/{id}/input.wav (v1.0); v1.5 lives under data/v1.5/")
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--version", default="both", choices=["1.0", "1.5", "both"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--servers_json", default=None,
                        help="Path to the JSON written by launch_moshi_servers.py --output-json")
    parser.add_argument("--server", nargs="*", default=None,
                        help="Explicit ws://… URLs (overrides --servers_json)")
    # Provenance-only: the server controls these, we just record what was used.
    parser.add_argument("--hf_repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--cfg_coef", default=None, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_name", default="steerduplex",
                        help="Label stored in run_meta.json for provenance")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files into mirror instead of symlinking")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate outputs even if they already exist")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    setup_logging()
    args = parse_args(argv)

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    servers = _load_servers(args.servers_json, args.server)

    logger.info("Dataset root: %s", dataset_root)
    logger.info("Output root:  %s", output_root)
    logger.info("Servers:      %s", servers)
    logger.info("Version: %s, tasks filter: %s", args.version, args.tasks)

    pending = _prepare_samples(
        dataset_root=dataset_root,
        output_root=output_root,
        version=args.version,
        tasks_filter=args.tasks,
        use_copy=args.copy,
        overwrite=args.overwrite,
    )
    logger.info("Total pending samples: %d", len(pending))

    if not pending:
        logger.info("Nothing to do.")
        _write_run_meta(output_root, args, servers, 0, {"ok": 0, "err": 0, "total": 0})
        return

    stats = run_inference(servers=servers, samples=pending, overwrite=args.overwrite)

    logger.info("Inference done. ok=%d err=%d total=%d",
                stats["ok"], stats["err"], stats["total"])
    _write_run_meta(output_root, args, servers, len(pending), stats)


if __name__ == "__main__":
    main()
