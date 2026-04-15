#!/usr/bin/env python3
"""
run_sessions.py — FDB v2 Revamped parallel session runner

Spawns N simultaneous examiner↔examinee conversations, shows a live Rich UI
with per-worker status, elapsed/ETA, and aggregates cost metadata.

Usage
  python run_sessions.py [--config configs/default.yaml]
"""

import argparse
import json
import os
import queue as _queue
import random
import signal
import socket
import subprocess
import sys
import threading

from dotenv import load_dotenv
load_dotenv()  # loads .env from cwd (or any parent) into os.environ
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional

import yaml
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Worker state (one per parallel slot, updated from worker threads)
# ---------------------------------------------------------------------------

@dataclass
class WorkerState:
    worker_id:   int
    task_id:     str             = ""
    split:       str             = ""
    status:      str             = "idle"   # idle | starting | running | done | failed
    start_time:  Optional[float] = None
    end_time:    Optional[float] = None
    cost_usd:    Optional[float] = None
    server_url:  str             = ""       # non-empty when using a pool-based adapter

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def reset(self):
        self.task_id    = ""
        self.split      = ""
        self.status     = "idle"
        self.start_time = None
        self.end_time   = None
        self.cost_usd   = None
        self.server_url = ""


# ---------------------------------------------------------------------------
# Server pool (generic URL pool for self-hosted adapters, e.g. Moshi)
# ---------------------------------------------------------------------------

class ServerPool:
    """
    Thread-safe pool of server URLs.

    Workers call acquire() before starting a conversation to get an exclusive
    URL, and release(url) when done so the next worker can reuse it.
    Each server handles one connection at a time, so the pool prevents multiple
    workers from hitting the same instance simultaneously.
    """

    def __init__(self, urls: list[str]):
        self._q: _queue.Queue[str] = _queue.Queue()
        for url in urls:
            self._q.put(url)

    def acquire(self) -> str:
        """Block until a server URL is available and return it."""
        return self._q.get()

    def release(self, url: str) -> None:
        """Return a URL to the pool."""
        self._q.put(url)


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------

def find_free_port(lo: int = 9000, hi: int = 15000) -> int:
    while True:
        port = random.randint(lo, hi)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------

def _adapter_cmd(script: str, static_args: dict, runtime_args: dict) -> list[str]:
    """
    Build a ['node', script, '--key', 'val', ...] command list.

    static_args come from the config YAML; runtime_args are injected by the
    runner (signalUrl, tokenServer, litellmBaseUrl, acquired pool URL, etc.).
    runtime_args override static_args when keys collide.
    """
    cmd = ["node", script]
    merged = {**static_args, **runtime_args}
    for k, v in merged.items():
        cmd += [f"--{k}", str(v)]
    return cmd


# ---------------------------------------------------------------------------
# Single conversation worker
# ---------------------------------------------------------------------------

def run_conversation(
    task:              dict,
    config:            dict,
    worker_state:      WorkerState,
    revamp_dir:        Path,
    token_server_port: int,
    server_pool:       "ServerPool | None" = None,
) -> dict:
    """
    Runs one full conversation session.
    Spawns orchestrator + examiner adapter + examinee adapter as subprocesses,
    waits for the configured duration, triggers graceful shutdown (SIGINT → ffmpeg
    mixdown), then reads back cost metadata written by the GPT adapter.
    """
    task_id = task["id"]
    split   = task["split"]

    worker_state.task_id    = task_id
    worker_state.split      = split
    worker_state.status     = "starting"
    worker_state.start_time = time.time()
    worker_state.end_time   = None
    worker_state.cost_usd   = None

    output_dir = Path(config["output_dir"]) / split / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    orch_port = find_free_port()

    env = {
        **os.environ,
        "RECORD_DIR":   str(output_dir),
        "SIGNAL_PORT":  str(orch_port),
        # Pass LiteLLM creds through env so Node processes can pick them up
        "LITELLM_BASE_URL": config["litellm"]["base_url"],
        "LITELLM_API_KEY":  os.environ["LITELLM_API_KEY"],
    }

    litellm_base = config["litellm"]["base_url"]
    litellm_key  = os.environ.get("LITELLM_API_KEY", "")
    signal_url   = f"ws://localhost:{orch_port}/signal"
    token_url    = f"http://localhost:{token_server_port}"
    duration     = int(config.get("conversation_duration", 120))

    # Merge examiner system + task prompts
    examiner_prompt = "\n\n".join(filter(None, [
        task.get("examiner_system_prompt", ""),
        task.get("examiner_task_prompt", ""),
    ]))

    examiner_cfg = config["examiner_adapter"]
    examinee_cfg = config["examinee_adapter"]

    procs: dict[str, subprocess.Popen] = {}
    pool_url: str | None = None
    result = {"task_id": task_id, "split": split, "status": "failed", "output_dir": str(output_dir)}

    try:
        # ── Orchestrator ────────────────────────────────────────────────────
        procs["orch"] = subprocess.Popen(
            ["node", "orchestrator.js"],
            cwd=str(revamp_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.8)  # let the signaling socket bind

        worker_state.status = "running"

        # ── Examiner (Role A) ────────────────────────────────────────────────
        # Always injects: role, signalUrl, tokenServer, litellmBaseUrl, systemPrompt.
        # Static args from config override defaults; runtime args take final precedence.
        examiner_runtime = {
            "role":           "A",
            "signalUrl":      signal_url,
            "litellmBaseUrl": litellm_base,
            "litellmApiKey":  litellm_key,
            "systemPrompt":   examiner_prompt,
        }
        procs["examiner"] = subprocess.Popen(
            _adapter_cmd(examiner_cfg["script"], examiner_cfg.get("args", {}), examiner_runtime),
            cwd=str(revamp_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # ── Examinee (Role B) ────────────────────────────────────────────────
        # Pool mode: acquire a server URL and inject it as --{url_arg}.
        # API mode (no pool key): inject litellm creds so the adapter can reach LiteLLM.
        if "pool" in examinee_cfg:
            pool_url = server_pool.acquire()  # type: ignore[union-attr]
            worker_state.server_url = pool_url
            examinee_runtime = {
                "role":      "B",
                "signalUrl": signal_url,
                examinee_cfg["pool"]["url_arg"]: pool_url,
            }
        else:
            examinee_runtime = {
                "role":           "B",
                "signalUrl":      signal_url,
                "litellmBaseUrl": litellm_base,
                "litellmApiKey":  litellm_key,
            }
        procs["examinee"] = subprocess.Popen(
            _adapter_cmd(examinee_cfg["script"], examinee_cfg.get("args", {}), examinee_runtime),
            cwd=str(revamp_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # ── Run for configured duration ──────────────────────────────────────
        deadline = time.time() + duration
        while time.time() < deadline:
            # Check if any critical process died early
            if procs["orch"].poll() is not None:
                break
            time.sleep(0.5)

        # ── Graceful shutdown: SIGTERM examiner first so it can write cost_A.json
        # before we tear down the orchestrator. SIGKILL (proc.kill) cannot be
        # caught by Node, so cost metadata would never be written.
        try:
            procs["examiner"].send_signal(signal.SIGTERM)
            procs["examiner"].wait(timeout=5)
        except Exception:
            pass

        # ── Trigger ffmpeg mixdown via SIGINT on the orchestrator ────────────
        try:
            procs["orch"].send_signal(signal.SIGINT)
            procs["orch"].wait(timeout=15)
        except Exception:
            procs["orch"].kill()

        # ── Read cost metadata written by the GPT adapter ───────────────────
        cost_file = output_dir / "cost_A.json"
        cost_usd  = None
        if cost_file.exists():
            with open(cost_file) as f:
                cost_data = json.load(f)
            cost_usd = cost_data.get("estimated_cost_usd")

        worker_state.cost_usd = cost_usd
        worker_state.status   = "done"
        result.update({"status": "done", "cost_usd": cost_usd})

    except Exception as exc:
        worker_state.status = "failed"
        result["error"] = str(exc)

    finally:
        worker_state.end_time = time.time()
        for proc in procs.values():
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        # Return pool URL so the next worker can use this server
        if pool_url is not None and server_pool is not None:
            server_pool.release(pool_url)

    return result


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------

STATUS_STYLE = {
    "idle":     ("dim",    "Idle"),
    "starting": ("yellow", "Starting…"),
    "running":  ("green",  "Running"),
    "done":     ("cyan",   "Done ✓"),
    "failed":   ("red bold", "Failed ✗"),
}

def build_display(
    workers:      list[WorkerState],
    total:        int,
    done:         int,
    wall_start:   float,
    progress_bar: Progress,
) -> Group:
    elapsed_total = time.time() - wall_start
    elapsed_str   = str(timedelta(seconds=int(elapsed_total)))

    if done > 0:
        eta_sec = (elapsed_total / done) * (total - done)
        eta_str = str(timedelta(seconds=int(eta_sec)))
    else:
        eta_str = "—"

    # ── Header ───────────────────────────────────────────────────────────
    header = Panel(
        f"[bold]Sessions:[/bold] {done}/{total}    "
        f"[bold]Elapsed:[/bold] {elapsed_str}    "
        f"[bold]ETA:[/bold] {eta_str}",
        title="[bold white]Full Duplex Bench v2[/bold white]",
        box=box.HEAVY,
        padding=(0, 2),
    )

    # ── Workers table ────────────────────────────────────────────────────
    # Show "Server" column only when any worker is using a pool-based adapter
    show_server = any(w.server_url for w in workers)

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan", expand=True)
    table.add_column("W#",      width=4,  justify="center")
    table.add_column("Task",    min_width=28)
    table.add_column("Split",   width=14)
    table.add_column("Status",  width=12)
    table.add_column("Elapsed", width=10, justify="right")
    table.add_column("Cost",    width=10, justify="right")
    if show_server:
        table.add_column("Server", width=8, justify="right")

    for w in workers:
        style, label = STATUS_STYLE.get(w.status, ("white", w.status))
        elapsed_str_w = str(timedelta(seconds=int(w.elapsed)))
        cost_str      = f"${w.cost_usd:.5f}" if w.cost_usd is not None else "—"

        row = [
            str(w.worker_id),
            w.task_id or "—",
            w.split   or "—",
            f"[{style}]{label}[/{style}]",
            elapsed_str_w,
            cost_str,
        ]
        if show_server:
            # Show just the port number (last component before path) for brevity
            port_str = w.server_url.split(":")[-1].split("/")[0] if w.server_url else "—"
            row.append(f"[dim]:{port_str}[/dim]" if w.server_url else "—")
        table.add_row(*row)

    return Group(header, table, progress_bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FDB v2 Revamped — parallel session runner")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    revamp_dir  = Path(__file__).parent.resolve()
    config_path = revamp_dir / args.config

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Backwards-compat shim: old-style flat config → adapter blocks ─────
    # Existing configs that don't have examiner_adapter / examinee_adapter keys
    # are automatically migrated so run_sessions.py stays model-agnostic.
    if "examiner_adapter" not in config:
        config["examiner_adapter"] = {
            "script": "adapters/gptRealtime_adapter.js",
            "args": {
                "model":     config.get("models", {}).get("examiner",
                             "openai/gpt-4o-realtime-preview-2024-12-17"),
                "voice":     config.get("voices", {}).get("examiner", "echo"),
                "vadMode":   config.get("examiner_vad_mode", "slow"),
                "autostart": "true",
            },
        }
    if "examinee_adapter" not in config:
        config["examinee_adapter"] = {
            "script": "adapters/gemini_adapter.js",
            "args": {
                "model":        config.get("models", {}).get("examinee",
                                "gemini/gemini-2.0-flash-live-001"),
                "systemPrompt": config.get("examinee_system_prompt",
                                "You are a helpful AI assistant."),
            },
        }

    # ── Resolve LiteLLM API key from environment (never stored in config) ─
    litellm_key = os.environ.get("LITELLM_API_KEY", "")
    if not litellm_key:
        print(
            "Error: LITELLM_API_KEY is not set.\n"
            "Add it to your .env file or export it before running:\n"
            "  export LITELLM_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build server pool if the examinee adapter uses one ────────────────
    server_pool: ServerPool | None = None
    examinee_cfg = config.get("examinee_adapter", {})
    if "pool" in examinee_cfg:
        pool_urls = examinee_cfg["pool"].get("urls", [])
        if not pool_urls:
            print(
                "Error: examinee_adapter.pool.urls is empty.\n"
                "Launch servers with launch_moshi_servers.py and add their URLs to the config.",
                file=sys.stderr,
            )
            sys.exit(1)
        server_pool = ServerPool(pool_urls)

    # ── Load and sample tasks ─────────────────────────────────────────────
    prompts_path = revamp_dir / "prompts_staged_200.json"
    with open(prompts_path) as f:
        prompts_data = json.load(f)

    splits_to_run = config.get("splits", list(prompts_data["splits"].keys()))
    all_tasks: list[dict] = []
    for split_name in splits_to_run:
        split_data = prompts_data["splits"].get(split_name, {})
        for task in split_data.get("tasks", []):
            all_tasks.append({**task, "split": split_name})

    num_samples = config.get("num_samples", -1)
    if num_samples != -1 and num_samples < len(all_tasks):
        all_tasks = random.sample(all_tasks, num_samples)

    total_tasks       = len(all_tasks)
    parallel_sessions = config.get("parallel_sessions", 4)

    console = Console()
    console.print(
        f"\n[bold green]FDB v2 Revamped[/bold green]  "
        f"tasks=[cyan]{total_tasks}[/cyan]  "
        f"parallel=[cyan]{parallel_sessions}[/cyan]  "
        f"config=[dim]{config_path}[/dim]\n"
    )
    if server_pool is not None:
        pool_count = len(examinee_cfg["pool"].get("urls", []))
        console.print(f"[dim]Examinee server pool: {pool_count} instance(s)[/dim]\n")

    # ── Start shared token server ─────────────────────────────────────────
    token_port = find_free_port()
    token_env  = {
        **os.environ,
        "LITELLM_BASE_URL": config["litellm"]["base_url"],
        "LITELLM_API_KEY":  litellm_key,
    }
    token_proc = subprocess.Popen(
        ["node", "adapters/token_server.js", "--port", str(token_port)],
        cwd=str(revamp_dir),
        env=token_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)  # let it bind
    console.print(f"[dim]Token server started on port {token_port}[/dim]\n")

    # ── Shared state ──────────────────────────────────────────────────────
    workers    = [WorkerState(worker_id=i + 1) for i in range(parallel_sessions)]
    results    = []
    done_count = 0
    done_lock  = threading.Lock()
    task_queue = list(all_tasks)
    queue_lock = threading.Lock()
    queue_idx  = [0]
    wall_start = time.time()
    shutdown_event = threading.Event()

    # ── Progress bar (lives outside the table so it persists cleanly) ────
    progress_bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=50),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )
    prog_task = progress_bar.add_task("Progress", total=total_tasks)

    # ── Worker thread function ────────────────────────────────────────────
    def worker_thread(worker_idx: int):
        nonlocal done_count
        while not shutdown_event.is_set():
            with queue_lock:
                if queue_idx[0] >= len(task_queue):
                    break
                task = task_queue[queue_idx[0]]
                queue_idx[0] += 1

            res = run_conversation(task, config, workers[worker_idx], revamp_dir, token_port,
                                   server_pool=server_pool)

            with done_lock:
                done_count += 1
                results.append(res)
                progress_bar.advance(prog_task)

            workers[worker_idx].reset()

    # ── Launch worker threads ─────────────────────────────────────────────
    threads = []
    for i in range(parallel_sessions):
        t = threading.Thread(target=worker_thread, args=(i,), daemon=True)
        threads.append(t)
        t.start()

    # ── Live display loop ─────────────────────────────────────────────────
    try:
        with Live(console=console, refresh_per_second=2, transient=False) as live:
            while any(t.is_alive() for t in threads):
                live.update(build_display(workers, total_tasks, done_count, wall_start, progress_bar))
                time.sleep(0.5)
            # Final frame
            live.update(build_display(workers, total_tasks, done_count, wall_start, progress_bar))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — stopping workers...[/yellow]")
        shutdown_event.set()

    for t in threads:
        t.join(timeout=5)

    # ── Cleanup token server ──────────────────────────────────────────────
    try:
        token_proc.kill()
        token_proc.wait(timeout=5)
    except Exception:
        pass

    # ── Write run summary ─────────────────────────────────────────────────
    total_elapsed   = time.time() - wall_start
    done_results    = [r for r in results if r["status"] == "done"]
    failed_results  = [r for r in results if r["status"] != "done"]
    total_cost      = sum(r.get("cost_usd") or 0.0 for r in done_results)

    summary = {
        "config":            str(config_path),
        "total_tasks":       total_tasks,
        "completed":         len(done_results),
        "failed":            len(failed_results),
        "total_cost_usd":    round(total_cost, 6),
        "elapsed_seconds":   round(total_elapsed, 1),
        "results":           results,
    }

    from datetime import datetime
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir  = Path(config["output_dir"])
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"run_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[bold]Done.[/bold]  "
                  f"[cyan]{len(done_results)}[/cyan]/{total_tasks} completed  "
                  f"[red]{len(failed_results)}[/red] failed  "
                  f"Total GPT cost: [green]${total_cost:.5f}[/green]")
    console.print(f"Summary → [dim]{summary_path}[/dim]\n")


if __name__ == "__main__":
    main()
