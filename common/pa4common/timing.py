"""CUDA-synchronised timing and peak-memory helpers.

Rules implemented here (assignment, "Shared Experimental Conventions"):
  * the measured path is warmed up before timing,
  * CUDA is synchronised before the clock starts and again before it stops,
  * repetitions, unit and device are always reported with the numbers.
"""
from __future__ import annotations

import contextlib
import statistics
import time
from typing import Callable

import torch


def _is_cuda(device) -> bool:
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def sync(device) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize(torch.device(device))


class Stopwatch:
    """Wall-clock stopwatch that synchronises CUDA on both edges."""

    def __init__(self, device="cpu"):
        self.device = device
        self.elapsed_s = 0.0

    def __enter__(self):
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.elapsed_s = time.perf_counter() - self._t0
        return False


def summarize_ms(times_ms: list[float]) -> dict:
    return {
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def time_callable(fn: Callable[[], object], *, device="cpu", warmup: int = 2, reps: int = 5) -> dict:
    """Time ``fn`` with warm-up and CUDA synchronisation. Unit: milliseconds."""
    assert reps >= 1
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    out = {"unit": "ms", "reps": reps, "warmup": warmup, "device": str(device), "times_ms": times}
    out.update(summarize_ms(times))
    return out


@contextlib.contextmanager
def peak_memory(device="cpu"):
    """Records ``torch.cuda.max_memory_allocated`` (bytes) for the enclosed block.

    This is *allocated* memory, not reserved memory and not packed/logical storage.
    On CPU the recorded value is ``None``.
    """
    rec = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    if _is_cuda(device):
        dev = torch.device(device)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    try:
        yield rec
    finally:
        if _is_cuda(device):
            dev = torch.device(device)
            torch.cuda.synchronize(dev)
            rec["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(dev))
            rec["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(dev))
