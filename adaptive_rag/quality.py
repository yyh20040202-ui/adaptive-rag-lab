"""V2: validation-selected cross-encoder, then revised-system test comparison."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .cross_encoder import CrossEncoder, blend_order
from .data import load_benchmark, file_sha256
from .metrics import aggregate, query_metrics, paired_bootstrap
from .retrieval import BM25, reciprocal_rank_fusion, top_indices
from .resources import CPU_THREADS, HEAVY_JOB_LOCK, ResourceRecorder
from .reranker_client import create_reranker

MAX_DEPTH = 100
DEPTHS = [20, 50, 100]
ALPHAS = [0., .25, .5, .75, 1.]


def validate_model_files(path, provenance):
    for record in provenance.get("files", [provenance.get("file")]):
        if file_sha256(path / record["path"]) != record["sha256"]:
            raise ValueError("Cross-encoder files changed; restore the pinned model before using score caches")


def evaluate_live(data, split, fingerprint, started):
    """Evaluate the exact serving engine once; retain all per-query outputs."""
    from .engine import RetrievalEngine
    selected = json.loads(Path("checkpoints/quality_config.json").read_text(encoding="utf-8"))
    if selected["embedding_fingerprint"] != fingerprint:
        raise ValueError("Embedding inputs changed after validation selection")
    path = Path(selected["model_path"])
    provenance = json.loads((path/"provenance.json").read_text(encoding="utf-8"))
    validate_model_files(path, provenance)
    if provenance["revision"] != selected["revision"] or provenance["model"] != selected["model"]:
        raise ValueError("Cross-encoder changed after validation selection")
    engine = RetrievalEngine.from_corpus("scifact")
    modes = ["bm25", "hybrid_lora", "neural_rerank", "adaptive", "cross_encoder"]
    # Warm on a training query, not a test query. Startup is excluded.
    for mode in modes:
        engine.search(data.queries[split["train"][0]], mode, 10)
    prior_rows = [json.loads(line) for line in Path("reports/per_query.jsonl").read_text(encoding="utf-8").splitlines()]
    prior = {row["query_id"]: row for row in prior_rows}
    recorder = ResourceRecorder()
    chosen = set(np.random.default_rng(42).choice(split["test"], size=40, replace=False).tolist())
    timings = {mode: [] for mode in modes}
    details, mismatches = [], []
    for index, qid in enumerate(split["test"]):
        recorder.sample()
        row = {"query_id": qid, "top10": {}, "metrics": {}}
        rotated = modes[index % len(modes):] + modes[:index % len(modes)]
        for mode in rotated:
            result = engine.search(data.queries[qid], mode, 10)
            ranking = [hit["source_id"] for hit in result["hits"]]
            row["top10"][mode] = ranking
            row["metrics"][mode] = query_metrics(ranking, data.test_qrels[qid])
            if mode == "adaptive":
                row["route"] = result["route"]
            if qid in chosen:
                timings[mode].append(result["retrieval_ms"])
            if mode != "cross_encoder" and ranking != prior[qid]["top10"][mode]:
                mismatches.append({"query_id": qid, "method": mode, "actual": ranking, "v1": prior[qid]["top10"][mode]})
        details.append(row)
        if index % 10 == 0:
            print(f"Evaluate live serving engine: {index+1}/300", flush=True)
    metrics = json.loads(Path("reports/retrieval.json").read_text(encoding="utf-8"))["metrics"]
    metrics.update({mode: aggregate([row["metrics"][mode] for row in details]) for mode in modes})
    comparisons = {}
    for baseline in ["bm25", "neural_rerank"]:
        a = np.array([row["metrics"]["cross_encoder"]["ndcg@10"] for row in details])
        b = np.array([row["metrics"][baseline]["ndcg@10"] for row in details])
        comparisons[f"cross_encoder-vs-{baseline}"] = paired_bootstrap(a, b, 42)
    latency = {mode: {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                      "mean_ms": float(np.mean(values)), "queries": len(values)} for mode, values in timings.items()}
    Path("reports/quality_per_query.jsonl").write_text("\n".join(json.dumps(row) for row in details)+"\n", encoding="utf-8")
    verification = {"queries": len(details), "live_ranking_evaluations": len(details)*len(modes),
                    "v1_ranking_comparisons": len(details)*4, "v1_mismatches": mismatches,
                    "v1_rankings_preserved": not mismatches,
                    "scope": "Actual serving engine generates V2 evaluation outputs; four legacy modes compared against V1 committed rankings; CE is evaluated through live engine, not independently rerun."}
    Path("reports/reference_verification_v2.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    report = {"dataset": "BEIR SciFact", "corpus": len(data.documents), "train": len(split["train"]),
              "validation": len(split["validation"]), "test": len(details), "primary_method": "cross_encoder", "seed": 42,
              "route_counts": {route: sum(row.get("route") == route for row in details) for route in ["bm25", "hybrid_lora", "neural_rerank"]},
              "dense_call_fraction": sum(row.get("route") != "bm25" for row in details)/len(details),
              "metrics": metrics, "latency": latency, "config": selected, "paired_ndcg_comparisons": comparisons,
              "resources": recorder.result(), "runtime_seconds": time.perf_counter()-started,
              "timing_protocol": "40 seed42 sampled test queries from300 live-engine pass; warmed serial requests; CPU query encoding included; selected CE backend recorded in config; mode order rotates; startup,indexing,generation excluded; 2 CPU threads; other system apps uncontrolled",
              "environment": {"cpu_threads": CPU_THREADS, "platform": platform.platform(), "python": platform.python_version()},
              "limitations": ["V2 iteration on previously inspected BEIR test; config chosen on validation only, not a pristine blind test",
                              "English biomedical retrieval; pretrained contamination unknown", "nDCG is ranking quality, not QA correctness",
                              "Cross-encoder uses upstream pretrained weights and quantized inference; no project CE fine-tuning",
                              "Unserved frozen/dense metrics retained from V1; five served methods re-evaluated in V2"]}
    Path("reports/quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics["cross_encoder"], "latency": latency, "resources": report["resources"], "v1_mismatches": len(mismatches)}, indent=2), flush=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(stage="select", variants=("l6", "l12")):
    started = time.perf_counter()
    from .cache import make_embedding_fingerprint, load_embedding_cache, save_embedding_cache
    from .encoder import Encoder
    data = load_benchmark()
    split = json.loads(Path("reports/split.json").read_text(encoding="utf-8"))
    for name, expected in split["input_sha256"].items():
        if file_sha256(Path("data/raw/scifact") / name) != expected:
            raise ValueError("Benchmark inputs changed after training/split; regenerate the experiment")
    ids = [d.id for d in data.documents]
    texts = [d.content for d in data.documents]
    query_ids = sorted(data.queries)
    queries = [data.queries[q] for q in query_ids]
    qp = {q: i for i, q in enumerate(query_ids)}
    fp = make_embedding_fingerprint(data.documents, data.queries, query_ids, max_length=192,
                                   adapter_path=Path("checkpoints/retriever_lora"))
    cache_path = Path("artifacts/lora_embeddings.npz")
    arrays = load_embedding_cache(cache_path, fp)
    if arrays is None:
        encoder = Encoder(adapter="checkpoints/retriever_lora", max_length=192, threads=CPU_THREADS)
        docs = encoder.encode(texts, batch_size=8, progress=True)
        qs = encoder.encode(queries, batch_size=8, progress=True)
        save_embedding_cache(cache_path, docs, qs, ids, query_ids, fp)
        del encoder
        gc.collect()
    else:
        docs, qs = arrays
    if stage == "evaluate":
        return evaluate_live(data, split, fp, started)
    bm25 = BM25(texts)
    qids = split["validation"]
    candidates, fusions = {}, {}
    for qid in qids:
        sp = bm25.scores(data.queries[qid])
        ds = docs @ qs[qp[qid]]
        fused = reciprocal_rank_fusion([top_indices(sp, MAX_DEPTH), top_indices(ds, MAX_DEPTH)], len(ids), 60, MAX_DEPTH)
        candidates[qid] = top_indices(fused, MAX_DEPTH)
        fusions[qid] = fused
    report_path = Path("reports/quality_selection.json")
    selected = None
    selection = []
    results = {}
    for variant in variants:
        path = Path("models/bge_reranker" if variant == "bge" else f"models/cross_encoder_{variant}")
        provenance = json.loads((path/"provenance.json").read_text(encoding="utf-8"))
        validate_model_files(path, provenance)
        inference = "GGUF-Q8_0-Vulkan" if variant == "bge" else "dynamic-int8-linear"
        model_config = {"model_path": str(path).replace("\\", "/"), "max_length": 384,
                        "backend": "llama_vulkan" if variant == "bge" else "torch_cpu"}
        if selected is not None and (provenance["model"] != selected["model"] or provenance["revision"] != selected["revision"]):
            raise ValueError("Cross-encoder version changed after validation selection")
        scoring_depth = MAX_DEPTH if stage == "select" else selected["rerank_depth"]
        signature = digest({"embedding": fp, "query_ids": qids, "candidates": {q: [ids[i] for i in candidates[q][:scoring_depth]] for q in qids},
                            "model": provenance, "max_length": 384, "inference": inference})
        score_path = Path(f"artifacts/ce_{variant}_{stage}.npz")
        cached_scores = None
        if score_path.exists():
            with np.load(score_path, allow_pickle=False) as cached:
                if str(cached["fingerprint"]) == signature:
                    cached_scores = cached["scores"]
                    if cached_scores.shape != (len(qids), scoring_depth) or cached_scores.dtype != np.float32 or not np.isfinite(cached_scores).all():
                        raise ValueError("Corrupt cross-encoder score cache")
        if cached_scores is not None:
            scores = cached_scores
            model = None
        else:
            model = create_reranker(model_config)
            scores = []
            for index, qid in enumerate(qids):
                scores.append(model.predict(data.queries[qid], [texts[i] for i in candidates[qid][:scoring_depth]]))
                if index % 10 == 0:
                    print(f"{stage} {variant}: {index+1}/{len(qids)} queries", flush=True)
            scores = np.stack(scores)
            np.savez_compressed(score_path, scores=scores, fingerprint=signature)
        if stage == "select":
            # Dynamic INT8 activation scaling depends on the batch. For each
            # prefix, recompute its last short batch instead of reusing logits
            # from a longer batch containing additional documents.
            depth_scores = {MAX_DEPTH: scores}
            for depth in DEPTHS:
                if depth == MAX_DEPTH:
                    continue
                prefix_path = Path(f"artifacts/ce_{variant}_prefix_{depth}.npz")
                prefix = None
                if prefix_path.exists():
                    with np.load(prefix_path, allow_pickle=False) as arrays:
                        if str(arrays["fingerprint"]) == signature:
                            prefix = arrays["scores"]
                            if prefix.shape != (len(qids), depth) or not np.isfinite(prefix).all():
                                raise ValueError("Corrupt prefix cache")
                if prefix is None:
                    if model is None:
                        model = create_reranker(model_config)
                    prefix = scores[:, :depth].copy()
                    tail_start = depth//8*8
                    if tail_start < depth and variant != "bge":
                        for index, qid in enumerate(qids):
                            prefix[index, tail_start:] = model.predict(data.queries[qid], [texts[i] for i in candidates[qid][tail_start:depth]])
                    np.savez_compressed(prefix_path, scores=prefix, fingerprint=signature)
                depth_scores[depth] = prefix
            for depth in DEPTHS:
                for alpha in ALPHAS:
                    rows = []
                    for index, qid in enumerate(qids):
                        order, _ = blend_order(candidates[qid], depth_scores[depth][index], fusions[qid], alpha, depth)
                        rows.append(query_metrics([ids[i] for i in order[:10]], data.train_qrels[qid]))
                    selection.append({"variant": variant, "model_path": str(path).replace("\\", "/"),
                                      "model": provenance["model"], "revision": provenance["revision"],
                                      "rerank_depth": depth, "alpha": alpha, "max_length": 384,
                                      "inference": inference, "backend": model_config["backend"],
                                      "model_file": provenance.get("file", {}).get("path", "model.safetensors"),
                                      "candidate_depth": MAX_DEPTH, "rrf_constant": 60,
                                      "validation_metrics": aggregate(rows), "embedding_fingerprint": fp})
            resource = model.recorder.result() if model else {"cached_scores": True}
            results[variant] = {"resources": resource, "score_fingerprint": signature,
                                "full_scores_reused": cached_scores is not None,
                                "resource_scope": "This invocation only; CPU full score caches may be reused and prefix tails recomputed"}
        else:
            rows = []
            for index, qid in enumerate(qids):
                order, _ = blend_order(candidates[qid], scores[index], fusions[qid], selected["alpha"], selected["rerank_depth"])
                rows.append({"query_id": qid, "top10": {"cross_encoder": [ids[i] for i in order[:10]]},
                             "metrics": {"cross_encoder": query_metrics([ids[i] for i in order[:10]], data.test_qrels[qid])}})
            results[variant] = rows
        del model
        gc.collect()
    if stage == "select":
        # Stable tie preference: shallower depth, then L6. No test labels are used here.
        selected = max(selection, key=lambda row: (row["validation_metrics"]["ndcg@10"], -row["rerank_depth"], row["variant"] == "l6"))
        report = {"validation_queries": len(qids), "grid": selection, "selected": selected, "resources": results,
                  "selection_protocol": "163 document-grouped official train queries; nDCG@10 primary; depth and model ties prefer less work; no test labels used for config selection",
                  "seconds": time.perf_counter()-started}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        Path("checkpoints/quality_config.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        print(json.dumps({"selected": {k:v for k,v in selected.items() if k != "embedding_fingerprint"}, "seconds": report["seconds"]}, indent=2), flush=True)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "evaluate"], default="select")
    parser.add_argument("--variants", nargs="+", choices=["l6", "l12", "bge"], default=["l6", "l12"])
    args = parser.parse_args()
    with HEAVY_JOB_LOCK:
        run(args.stage, args.variants)
