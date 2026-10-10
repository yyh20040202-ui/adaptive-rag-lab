"""Conservative resource guards for the 16 GB development laptop."""
from __future__ import annotations
import os
from pathlib import Path
import threading
import time

CPU_THREADS = max(1, min(int(os.getenv("ADAPTIVE_CPU_THREADS", "2")), 4))
MIN_AVAILABLE_GIB = float(os.getenv("ADAPTIVE_MIN_AVAILABLE_GIB", "2.0"))
MAX_PROCESS_GIB = float(os.getenv("ADAPTIVE_MAX_PROCESS_GIB", "3.0"))
HEAVY_JOB_LOCK = threading.RLock()


class ResourcePressure(RuntimeError):
    pass


def snapshot() -> dict:
    try:
        import psutil
        memory = psutil.virtual_memory()
        rss = psutil.Process().memory_info().rss
        return {"available_gib": memory.available / 1024**3, "total_gib": memory.total / 1024**3,
                "memory_percent": memory.percent, "process_rss_gib": rss / 1024**3,
                "cpu_threads": CPU_THREADS, "min_available_gib": MIN_AVAILABLE_GIB,
                "max_process_gib": MAX_PROCESS_GIB, "max_pair_batch": 8,
                "heavy_jobs_serial": True, "serialization_scope": "within this process; launch CLI heavy jobs sequentially"}
    except ImportError:
        return {"cpu_threads": CPU_THREADS, "min_available_gib": MIN_AVAILABLE_GIB,
                "max_process_gib": MAX_PROCESS_GIB, "max_pair_batch": 8,
                "heavy_jobs_serial": True, "serialization_scope": "within this process", "memory_monitor_available": False}


def check_resources() -> dict:
    state = snapshot()
    if "available_gib" not in state:
        raise ResourcePressure("Install psutil to enable the required memory guard")
    if state["available_gib"] < MIN_AVAILABLE_GIB:
        raise ResourcePressure(f"Only {state['available_gib']:.2f} GiB available; reserve {MIN_AVAILABLE_GIB:.1f} GiB. Computation stopped safely.")
    if state["process_rss_gib"] > MAX_PROCESS_GIB:
        raise ResourcePressure("Process memory limit exceeded; computation stopped safely")
    return state


class ResourceRecorder:
    def __init__(self):
        self.started = time.perf_counter()
        self.max_rss_gib = 0.
        self.min_available_gib = float("inf")
        self.samples = 0

    def sample(self):
        state = check_resources()
        self.max_rss_gib = max(self.max_rss_gib, state["process_rss_gib"])
        self.min_available_gib = min(self.min_available_gib, state["available_gib"])
        self.samples += 1
        return state

    def result(self):
        return {"peak_observed_process_rss_gib": self.max_rss_gib,
                "minimum_observed_available_gib": self.min_available_gib,
                "samples": self.samples, "seconds": time.perf_counter()-self.started,
                "cpu_threads": CPU_THREADS, "batch_limit": 8,
                "minimum_available_gib_guard": MIN_AVAILABLE_GIB,
                "note": "Sampled before batches, not an OS-enforced hard allocation cap; other apps can change available RAM."}


if __name__ == "__main__":
    import json
    print(json.dumps(snapshot(), indent=2), flush=True)
