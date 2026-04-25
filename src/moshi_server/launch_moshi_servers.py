#!/usr/bin/env python3
"""
launch_moshi_servers.py — Launch N moshi_server.py instances across G GPUs.

Spreads instances round-robin across the provided GPU IDs, health-checks each
via WebSocket (waits for the 0x00 handshake byte), then stays running to keep
the child processes alive.

Usage:
    python launch_moshi_servers.py \\
        --num-instances 6 \\
        --gpu-ids 0,1,2 \\
        --base-port 9100 \\
        --hf-repo kyutai/moshi-2-0-80m \\
        [--moshi-weight /path/to/checkpoint] \\
        [--cfg-coef 1.0] \\
        [--output-json moshi_servers.json]

Press Ctrl+C to shut down all instances.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def wait_for_server(url: str, timeout: float = 300.0, poll_interval: float = 2.0) -> bool:
    """
    Poll the Moshi WebSocket endpoint until the server sends the 0x00 handshake
    byte, or until timeout seconds have elapsed.

    Returns True on success, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, timeout=aiohttp.ClientTimeout(total=5)) as ws:
                    msg = await asyncio.wait_for(ws.receive_bytes(), timeout=5.0)
                    if msg and msg[0] == 0x00:
                        return True
        except Exception:
            pass
        await asyncio.sleep(poll_interval)
    return False


async def health_check_all(servers: list[dict], timeout: float) -> None:
    """Check all servers concurrently, printing status as each becomes ready."""
    async def check_one(info: dict) -> None:
        url = info["url"]
        gpu = info["gpu"]
        port = info["port"]
        ok = await wait_for_server(url, timeout=timeout)
        if ok:
            print(f"  ✓  ws://localhost:{port}/api/chat  (GPU {gpu})", flush=True)
        else:
            print(f"  ✗  ws://localhost:{port}/api/chat  TIMEOUT after {timeout}s", flush=True)
            info["healthy"] = False
            return
        info["healthy"] = True

    await asyncio.gather(*[check_one(s) for s in servers])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch N moshi_server.py instances across G GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-instances", type=int, required=True,
                        help="Total number of server instances to launch.")
    parser.add_argument("--gpu-ids", type=str, required=True,
                        help="Comma-separated GPU IDs to spread instances across, e.g. '0,1,2'.")
    parser.add_argument("--base-port", type=int, default=9100,
                        help="First port to use; subsequent instances get base_port+1, +2, … (default: 9100).")
    parser.add_argument("--health-timeout", type=float, default=300.0,
                        help="Seconds to wait for each server to become healthy (default: 300).")

    # Passthrough args forwarded verbatim to moshi_server.py
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace repo for the model (forwarded to moshi_server.py).")
    parser.add_argument("--moshi-weight", type=str, default=None,
                        help="Path to a local Moshi checkpoint (forwarded to moshi_server.py).")
    parser.add_argument("--mimi-weight", type=str, default=None,
                        help="Path to a local Mimi checkpoint (forwarded to moshi_server.py).")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to a local tokenizer file (forwarded to moshi_server.py).")
    parser.add_argument("--cfg-coef", type=float, default=None,
                        help="CFG coefficient (forwarded to moshi_server.py).")
    parser.add_argument("--half", action="store_true", default=False,
                        help="Use float16 instead of bfloat16 (forwarded to moshi_server.py).")
    parser.add_argument("--text-prompt", type=str, default=None,
                        help="System prompt text, applied identically across all spawned "
                             "instances (forwarded to moshi_server.py).")
    parser.add_argument("--voice-prompt", type=str, default=None,
                        help="Path to voice prompt WAV or .pt, applied identically across "
                             "all spawned instances (forwarded to moshi_server.py).")

    parser.add_argument("--output-json", type=str, default=None,
                        help="If set, write the list of server URLs to this JSON file once all are healthy.")
    parser.add_argument("--server-script", type=str, default=None,
                        help="Override path to moshi_server.py (default: same directory as this script).")
    parser.add_argument("--log-dir", type=str, default="logs/moshi_servers",
                        help="Directory for per-instance log files (default: logs/moshi_servers).")

    args = parser.parse_args()

    gpu_list = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    if not gpu_list:
        print("Error: --gpu-ids must be a non-empty comma-separated list.", file=sys.stderr)
        sys.exit(1)

    server_script = Path(args.server_script) if args.server_script else Path(__file__).parent / "moshi_server.py"
    if not server_script.exists():
        print(f"Error: moshi_server.py not found at {server_script}", file=sys.stderr)
        sys.exit(1)

    # Build the static passthrough args for moshi_server.py
    passthrough: list[str] = ["--host", "localhost", "--static", "none"]
    if args.hf_repo:
        passthrough += ["--hf-repo", args.hf_repo]
    if args.moshi_weight:
        passthrough += ["--moshi-weight", args.moshi_weight]
    if args.mimi_weight:
        passthrough += ["--mimi-weight", args.mimi_weight]
    if args.tokenizer:
        passthrough += ["--tokenizer", args.tokenizer]
    if args.cfg_coef is not None:
        passthrough += ["--cfg-coef", str(args.cfg_coef)]
    if args.half:
        passthrough += ["--half"]
    if args.text_prompt is not None:
        passthrough += ["--text-prompt", args.text_prompt]
    if args.voice_prompt is not None:
        passthrough += ["--voice-prompt", args.voice_prompt]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    log_files: list = []  # keep file handles open for the lifetime of the launcher
    servers: list[dict] = []

    # ── Graceful shutdown ────────────────────────────────────────────────────
    def shutdown(signum=None, frame=None):
        print("\nShutting down Moshi servers…", flush=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        for fh in log_files:
            try:
                fh.close()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Spawn all instances ──────────────────────────────────────────────────
    print(f"\nLaunching {args.num_instances} Moshi server(s) across GPU(s) {args.gpu_ids}…", flush=True)
    print(f"Logs → {log_dir.resolve()}/\n", flush=True)
    for i in range(args.num_instances):
        gpu = gpu_list[i % len(gpu_list)]
        port = args.base_port + i
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}

        log_path = log_dir / f"moshi_{port}.log"
        log_fh = open(log_path, "w", buffering=1)  # line-buffered so tail -f works
        log_files.append(log_fh)

        cmd = [
            sys.executable, str(server_script),
            "--port", str(port),
            *passthrough,
        ]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=log_fh,
        )
        procs.append(proc)
        servers.append({"url": f"ws://localhost:{port}/api/chat", "port": port, "gpu": gpu, "healthy": False})
        print(f"  spawned PID {proc.pid}  port={port}  GPU={gpu}  log={log_path.name}", flush=True)

    print(f"\nWaiting for all {args.num_instances} server(s) to become healthy"
          f" (timeout: {args.health_timeout:.0f}s each)…\n", flush=True)

    asyncio.run(health_check_all(servers, timeout=args.health_timeout))

    healthy = [s for s in servers if s.get("healthy")]
    unhealthy = [s for s in servers if not s.get("healthy")]

    print(f"\n{len(healthy)}/{args.num_instances} server(s) healthy.", flush=True)
    if unhealthy:
        print(f"WARNING: {len(unhealthy)} server(s) failed to start:", file=sys.stderr)
        for s in unhealthy:
            print(f"  ws://localhost:{s['port']}/api/chat  (GPU {s['gpu']})", file=sys.stderr)

    urls = [s["url"] for s in servers]

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"servers": urls}, f, indent=2)
        print(f"\nServer URLs written to {out_path}", flush=True)

    print("\nAll servers running. Press Ctrl+C to shut down.\n", flush=True)

    # ── Keep alive ───────────────────────────────────────────────────────────
    while True:
        # Detect any crashed child and warn
        for i, (proc, info) in enumerate(zip(procs, servers)):
            if proc.poll() is not None:
                print(f"WARNING: server on port {info['port']} (GPU {info['gpu']}) exited "
                      f"with code {proc.returncode}.", file=sys.stderr, flush=True)
                # Replace entry so we don't spam the warning
                procs[i] = subprocess.Popen(["true"])  # sentinel — already dead
        time.sleep(10)


if __name__ == "__main__":
    main()
