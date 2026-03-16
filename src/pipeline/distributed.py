"""Distributed work coordination for multi-node, multi-process pipelines.

Uses atomic file creation (O_CREAT | O_EXCL) on the shared filesystem (EFS/NFS)
to ensure no two workers claim the same work item, regardless of which node or
process they run on.

No external services needed — just a shared filesystem.

Usage:
    from pipeline.distributed import try_claim, release_claim, is_done

    for item_id in work_items:
        if is_done(output_dir / f"{item_id}.json"):
            continue
        if not try_claim(claims_dir / f"{item_id}.claim"):
            continue  # another worker got it
        try:
            do_work(item_id)
        except Exception:
            release_claim(claims_dir / f"{item_id}.claim")  # let someone else retry
"""

import os
import socket
import time
from pathlib import Path
from typing import NamedTuple


# Stale claim timeout: if a claim file is older than this, assume the worker
# died and allow re-claiming. Set conservatively high (2 hours) — a single
# conversation should never take this long.
STALE_CLAIM_TIMEOUT_SEC = 7200


def try_claim(claim_path: Path) -> bool:
    """Atomically claim a work item. Returns True if successfully claimed.

    Works across nodes on NFS/EFS because O_CREAT | O_EXCL is atomic
    at the filesystem level.
    """
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        info = f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}"
        os.write(fd, info.encode())
        os.close(fd)
        return True
    except FileExistsError:
        # Check if it's a stale claim from a dead worker
        try:
            age = time.time() - os.path.getmtime(str(claim_path))
            if age > STALE_CLAIM_TIMEOUT_SEC:
                os.unlink(str(claim_path))
                # Try again after removing stale claim
                return try_claim(claim_path)
        except OSError:
            pass
        return False


def release_claim(claim_path: Path):
    """Release a claim (e.g., on failure) so another worker can retry."""
    try:
        os.unlink(str(claim_path))
    except OSError:
        pass


def is_done(output_path: Path) -> bool:
    """Check if work item is already completed."""
    return output_path.exists()


def mark_done(claim_path: Path):
    """Remove claim file after output is written (optional cleanup)."""
    release_claim(claim_path)


# ---------------------------------------------------------------------------
# GPU memory-aware worker planning
# ---------------------------------------------------------------------------
class GPUInfo(NamedTuple):
    gpu_id: int
    free_mb: int
    total_mb: int


def get_gpu_info() -> list[GPUInfo]:
    """Get free and total memory for each visible CUDA GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        infos = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            infos.append(GPUInfo(
                gpu_id=i,
                free_mb=int(free / 1024 / 1024),
                total_mb=int(total / 1024 / 1024),
            ))
        return infos
    except Exception:
        return []


class WorkerPlan(NamedTuple):
    gpu_id: int
    num_workers: int


def plan_workers(
    mem_per_worker_mb: int,
    max_workers_per_gpu: int,
    min_free_after_mb: int = 4096,
    num_gpus: int | None = None,
    reserve_fraction: float = 0.10,
) -> list[WorkerPlan]:
    """Plan how many workers to spawn per GPU based on available memory.

    Uses conservative estimates: the memory check happens at launch but
    KV cache and PyTorch allocator fragmentation grow during inference.

    Args:
        mem_per_worker_mb: Estimated VRAM per worker (model weights + KV cache peak).
        max_workers_per_gpu: Upper bound on workers per GPU.
        min_free_after_mb: Minimum free VRAM to leave after all workers spawn.
        num_gpus: Limit to this many GPUs (None = all available).
        reserve_fraction: Additionally reserve this fraction of total VRAM
            as headroom for non-PyTorch allocations (CUDA context, cuDNN, etc).

    Returns:
        List of (gpu_id, num_workers) for GPUs with enough memory.
        GPUs with insufficient memory are skipped entirely.
    """
    gpus = get_gpu_info()
    if not gpus:
        return []

    if num_gpus is not None:
        gpus = gpus[:num_gpus]

    plans = []
    for gpu in gpus:
        # Reserve a fraction of total VRAM for CUDA overhead + other processes
        reserved = max(min_free_after_mb, int(gpu.total_mb * reserve_fraction))
        usable = gpu.free_mb - reserved
        if usable < mem_per_worker_mb:
            print(f"  [GPU {gpu.gpu_id}] SKIP — {gpu.free_mb}MB free, "
                  f"{reserved}MB reserved, need {mem_per_worker_mb}MB/worker")
            continue
        n = min(usable // mem_per_worker_mb, max_workers_per_gpu)
        if n > 0:
            total_used = n * mem_per_worker_mb
            headroom = gpu.free_mb - total_used - reserved
            plans.append(WorkerPlan(gpu_id=gpu.gpu_id, num_workers=n))
            print(f"  [GPU {gpu.gpu_id}] {n} workers × {mem_per_worker_mb}MB = {total_used}MB "
                  f"({gpu.free_mb}MB free, {headroom}MB headroom)")

    return plans
