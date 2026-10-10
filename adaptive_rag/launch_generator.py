"""Run one owned llama.cpp child with a RAM watchdog; Ctrl+C stops that child."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .resources import CPU_THREADS, MIN_AVAILABLE_GIB, MAX_PROCESS_GIB, check_resources, ResourcePressure
from .data import file_sha256


def run(model_directory="models/qwen17", gpu_layers=99, reranker=False, context=3072, no_mmap=False):
    import psutil
    check_resources()
    directory = Path(model_directory)
    provenance = json.loads((directory/"provenance.json").read_text(encoding="utf-8"))
    filename = provenance.get("file", {}).get("path")
    if filename is None:
        filename = next(item["path"] for item in provenance["files"] if item["path"].endswith(".gguf"))
    executable = Path(os.getenv("LLAMA_SERVER_EXE", "models/llama/llama-server.exe")).resolve()
    model = (directory/filename).resolve()
    if not executable.is_file() or not model.is_file():
        raise FileNotFoundError("Download the pinned GGUF and llama.cpp runtime first")
    record = provenance.get("file") or next(item for item in provenance["files"] if item["path"] == filename)
    if model.stat().st_size != record["bytes"] or file_sha256(model) != record["sha256"]:
        raise ValueError("GGUF size/hash differs from pinned download; refusing to start")
    command = [str(executable), "-m", str(model), "--host", "127.0.0.1", "--port", "8089" if reranker else "8087",
               "--ctx-size", "512" if reranker else str(context), "--parallel", "1", "--n-gpu-layers", str(gpu_layers),
               "--threads", str(CPU_THREADS), "--threads-batch", str(CPU_THREADS),
               "--batch-size", "512" if reranker else "256", "--ubatch-size", "512" if reranker else "128",
               "--cache-ram", "0", "--ctx-checkpoints", "0"]
    command += ["--embedding", "--pooling", "rank", "--reranking"] if reranker else ["--jinja", "--reasoning", "off"]
    if no_mmap:
        command += ["--load-mode", "none"]  # Pinned b11517 replaces the old --no-mmap switch.
    Path("artifacts").mkdir(exist_ok=True)
    log_path = Path("artifacts/reranker.log" if reranker else "artifacts/generator.log")
    print(f"Starting {filename}; RAM watchdog reserves {MIN_AVAILABLE_GIB} GiB. Log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while child.poll() is None:
                time.sleep(2)
                available = psutil.virtual_memory().available/1024**3
                try:
                    rss = psutil.Process(child.pid).memory_info().rss/1024**3
                except psutil.NoSuchProcess:
                    break
                if available < MIN_AVAILABLE_GIB or rss > MAX_PROCESS_GIB:
                    raise ResourcePressure(f"Generator stopped: available {available:.2f} GiB; child RSS {rss:.2f} GiB")
            if child.returncode not in (None, 0):
                raise RuntimeError("llama.cpp exited; inspect artifacts/generator.log")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", default="models/qwen17")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--context", type=int, choices=[2048, 3072, 4096], default=3072)
    parser.add_argument("--no-mmap", action="store_true", help="Use direct weight loading; remeasure RAM and VRAM on your hardware")
    args = parser.parse_args()
    run(args.model_directory, args.gpu_layers, args.reranker, args.context, args.no_mmap)
