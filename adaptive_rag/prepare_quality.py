"""Pinned safe-tensor cross-encoders and optional larger local Qwen GGUF."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .prepare import fetch, get_json

REVISIONS = {
    "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "cross-encoder/ms-marco-MiniLM-L12-v2": "7b0235231ca2674cb8ca8f022859a6eba2b1c968",
    "ggml-org/Qwen3-1.7B-GGUF": "daeb8e2d528a760970442092f6bf1e55c3b659eb",
    "lmstudio-community/Qwen3-4B-Instruct-2507-GGUF": "4edb920b6f14e3b9284d4502a6485103d72cde05",
    "gpustack/bge-reranker-v2-m3-GGUF": "3093af03b1a635e67b084b1d8c03c5f5e020fd05",
}


def model_metadata(model_id: str):
    revision = REVISIONS[model_id]
    tree = get_json(f"https://huggingface.co/api/models/{model_id}/tree/{revision}")
    return revision, {row["path"]: row for row in tree}


def download_cross_encoder(model_id: str, destination: Path):
    revision, files = model_metadata(model_id)
    if "model.safetensors" not in files:
        raise ValueError("Refusing pickle-only model; a safetensors revision is required")
    names = [name for name in ["config.json", "tokenizer_config.json", "special_tokens_map.json",
                               "tokenizer.json", "vocab.txt", "model.safetensors"] if name in files]
    records = [fetch(f"https://huggingface.co/{model_id}/resolve/{revision}/{name}?download=true",
                     destination/name, files[name].get("lfs", {}).get("oid"), files[name].get("size")) for name in names]
    provenance = {"model": model_id, "revision": revision, "license": "Apache-2.0", "files": records}
    (destination/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"model": model_id, "revision": revision, "bytes": sum(r["bytes"] for r in records)}, indent=2), flush=True)


def download_qwen_17(destination: Path):
    model_id = "ggml-org/Qwen3-1.7B-GGUF"
    revision, files = model_metadata(model_id)
    # Prefer the current 4-bit K quantization; this publisher may only expose Q4_0.
    matches = [name for name in files if name.endswith("Q4_K_M.gguf") or name.endswith("Q4_0.gguf")]
    if not matches:
        raise ValueError("No supported compact 4-bit GGUF in the pinned repository")
    name = sorted(matches, key=lambda item: (not item.endswith("Q4_K_M.gguf"), item))[0]
    result = fetch(f"https://huggingface.co/{model_id}/resolve/{revision}/{name}?download=true",
                   destination/name, files[name].get("lfs", {}).get("oid"), files[name].get("size"))
    metadata = {"model": model_id, "base_model": "Qwen/Qwen3-1.7B", "revision": revision,
                "license": "Apache-2.0", "file": result}
    (destination/"provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def download_gpu_reranker(destination: Path):
    model_id = "gpustack/bge-reranker-v2-m3-GGUF"
    revision, files = model_metadata(model_id)
    name = "bge-reranker-v2-m3-Q8_0.gguf"
    result = fetch(f"https://huggingface.co/{model_id}/resolve/{revision}/{name}?download=true",
                   destination/name, files[name]["lfs"]["oid"], files[name]["size"])
    metadata = {"model": model_id, "base_model": "BAAI/bge-reranker-v2-m3", "revision": revision,
                "license": "Apache-2.0", "file": result, "format": "Q8_0 GGUF", "publisher": "GPUStack conversion of BAAI model"}
    (destination/"provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def download_qwen_4(destination: Path):
    model_id = "lmstudio-community/Qwen3-4B-Instruct-2507-GGUF"
    revision, files = model_metadata(model_id)
    name = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    result = fetch(f"https://huggingface.co/{model_id}/resolve/{revision}/{name}?download=true",
                   destination/name, files[name]["lfs"]["oid"], files[name]["size"])
    metadata = {"model": model_id, "base_model": "Qwen/Qwen3-4B-Instruct-2507", "revision": revision,
                "license": "Apache-2.0", "file": result, "publisher": "LM Studio community conversion"}
    (destination/"provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", action="store_true")
    parser.add_argument("--generator-4b", action="store_true")
    parser.add_argument("--l12", action="store_true")
    parser.add_argument("--gpu-reranker", action="store_true")
    args = parser.parse_args()
    if args.generator_4b:
        download_qwen_4(Path("models/qwen4"))
    elif args.gpu_reranker:
        download_gpu_reranker(Path("models/bge_reranker"))
    elif args.generator:
        download_qwen_17(Path("models/qwen17"))
    else:
        download_cross_encoder("cross-encoder/ms-marco-MiniLM-L6-v2", Path("models/cross_encoder_l6"))
        if args.l12:
            download_cross_encoder("cross-encoder/ms-marco-MiniLM-L12-v2", Path("models/cross_encoder_l12"))
