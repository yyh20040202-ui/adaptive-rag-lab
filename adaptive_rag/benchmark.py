"""Validation-only configuration selection; one final official BEIR test pass."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

from .data import load_benchmark, file_sha256
from .cache import make_embedding_fingerprint, load_embedding_cache, save_embedding_cache
from .encoder import Encoder
from .metrics import aggregate, paired_bootstrap, query_metrics
from .ranking import fit_ranker, fit_router, save_head
from .retrieval import (BM25, normalized_scores, pair_features, reciprocal_rank_fusion,
                        routing_features, tokenize, top_indices)

ROUTES = ["bm25", "hybrid_lora", "neural_rerank"]
METHODS = ["bm25", "dense_frozen", "hybrid_frozen", "dense_lora", "hybrid_lora", "neural_rerank", "adaptive"]


def run(seed: int = 42, latency_queries: int = 40) -> None:
    start = time.perf_counter()
    from .resources import CPU_THREADS
    torch.set_num_threads(CPU_THREADS)
    benchmark = load_benchmark()
    split = json.loads(Path("reports/split.json").read_text(encoding="utf-8"))
    if split["seed"] != seed:
        raise ValueError("Train and evaluation seeds differ")
    ids = [d.id for d in benchmark.documents]
    positions = {did: i for i, did in enumerate(ids)}
    texts = [d.content for d in benchmark.documents]
    query_ids = sorted(benchmark.queries)
    query_positions = {qid: i for i, qid in enumerate(query_ids)}
    queries = [benchmark.queries[qid] for qid in query_ids]
    bm25 = BM25(texts)
    cache = Path("artifacts")
    cache.mkdir(exist_ok=True)
    encoder = Encoder(max_length=192)
    frozen_file = cache / "frozen_embeddings.npz"
    frozen_fingerprint = make_embedding_fingerprint(benchmark.documents, benchmark.queries, query_ids,
                                                    max_length=192, dimension=encoder.model.config.hidden_size)
    try:
        frozen_cached = load_embedding_cache(frozen_file, frozen_fingerprint)
    except ValueError as error:
        print(f"{error}; reindexing frozen MiniLM", flush=True)
        frozen_cached = None
    if frozen_cached is None:
        print("Indexing frozen MiniLM", flush=True)
        frozen_docs = encoder.encode(texts, progress=True)
        frozen_queries = encoder.encode(queries, progress=True)
        save_embedding_cache(frozen_file, frozen_docs, frozen_queries, ids, query_ids, frozen_fingerprint)
    else:
        frozen_docs, frozen_queries = frozen_cached
    del encoder
    encoder = Encoder(adapter="checkpoints/retriever_lora", max_length=192)
    lora_file = cache / "lora_embeddings.npz"
    adapter_hash = file_sha256(Path("checkpoints/retriever_lora/adapter_model.safetensors"))
    lora_fingerprint = make_embedding_fingerprint(benchmark.documents, benchmark.queries, query_ids,
                                                 adapter_path="checkpoints/retriever_lora", max_length=192,
                                                 dimension=encoder.model.config.hidden_size)
    try:
        lora_cached = load_embedding_cache(lora_file, lora_fingerprint)
    except ValueError as error:
        print(f"{error}; reindexing LoRA MiniLM", flush=True)
        lora_cached = None
    if lora_cached is None:
        print("Indexing LoRA MiniLM", flush=True)
        lora_docs = encoder.encode(texts, progress=True)
        lora_queries = encoder.encode(queries, progress=True)
        save_embedding_cache(lora_file, lora_docs, lora_queries, ids, query_ids, lora_fingerprint)
    else:
        lora_docs, lora_queries = lora_cached
    if frozen_docs.shape != lora_docs.shape or len(frozen_docs) != len(ids):
        raise ValueError("Invalid embedding cache")
    all_sparse, all_dense, all_rankings, all_features = {}, {}, {}, {}
    all_frozen_dense = frozen_queries @ frozen_docs.T
    all_lora_dense = lora_queries @ lora_docs.T
    for qid in split["train"] + split["validation"] + split["test"]:
        row = query_positions[qid]
        sp = bm25.scores(benchmark.queries[qid])
        fd, ld = all_frozen_dense[row], all_lora_dense[row]
        br, fr, lr = top_indices(sp, 100), top_indices(fd, 100), top_indices(ld, 100)
        hf = reciprocal_rank_fusion([br, fr], len(ids))
        hl = reciprocal_rank_fusion([br, lr], len(ids))
        all_sparse[qid], all_dense[qid] = sp, ld
        all_rankings[qid] = {"bm25": br, "dense_frozen": fr, "dense_lora": lr,
                              "hybrid_frozen": top_indices(hf, 100), "hybrid_lora": top_indices(hl, 100)}
        all_features[qid] = routing_features(benchmark.queries[qid], sp, bm25)

    def features(qid: str, candidates: np.ndarray) -> np.ndarray:
        terms = set(tokenize(benchmark.queries[qid]))
        coverage = np.array([len(terms & bm25.text_tokens[i]) / max(len(terms), 1) for i in candidates])
        return pair_features(lora_queries[query_positions[qid]], lora_docs[candidates],
                              all_sparse[qid][candidates], all_dense[qid][candidates], coverage)

    def training_pairs(qids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for qid in qids:
            relevant = benchmark.train_qrels[qid]
            positives = [positions[did] for did in relevant]
            hard = [int(i) for i in all_rankings[qid]["hybrid_lora"] if ids[i] not in relevant][:7]
            candidates = np.array(positives + hard)
            x.append(features(qid, candidates))
            y.extend([1] * len(positives) + [0] * len(hard))
        return np.concatenate(x), np.array(y)

    print("Training neural learning-to-rank head", flush=True)
    tx, ty = training_pairs(split["train"])
    vx, vy = training_pairs(split["validation"])
    ranker, ranker_report = fit_ranker(tx, ty, vx, vy, seed)
    save_head(ranker, Path("checkpoints/ranker.safetensors"))
    neural_scores = {}
    fusion_scores = {}
    for qid in split["train"] + split["validation"] + split["test"]:
        candidates = all_rankings[qid]["hybrid_lora"]
        neural_scores[qid] = ranker.predict(features(qid, candidates))
        br, lr = all_rankings[qid]["bm25"], all_rankings[qid]["dense_lora"]
        fusion_scores[qid] = normalized_scores(reciprocal_rank_fusion([br, lr], len(ids))[candidates])

    def reranked(qid: str, alpha: float) -> np.ndarray:
        candidates = all_rankings[qid]["hybrid_lora"]
        values = (1-alpha) * fusion_scores[qid] + alpha * neural_scores[qid]
        return candidates[top_indices(values, len(candidates))]

    def score(qid: str, ranking: np.ndarray, qrels: dict) -> dict:
        return query_metrics([ids[i] for i in ranking[:10]], qrels[qid])

    alphas = []
    for alpha in [0., .15, .35, .65, 1.]:
        value = np.mean([score(qid, reranked(qid, alpha), benchmark.train_qrels)["ndcg@10"]
                         for qid in split["validation"]])
        alphas.append({"alpha": alpha, "validation_ndcg@10": float(value)})
    alpha = max(alphas, key=lambda row: row["validation_ndcg@10"])["alpha"]
    for qid in all_rankings:
        all_rankings[qid]["neural_rerank"] = reranked(qid, alpha)
    print(f"Selected neural blend alpha={alpha} on validation", flush=True)

    # Routing utility uses fixed proxy costs, never test labels or post-retrieval dense features.
    # Runtime evaluation below separately measures wall-clock latency.
    costs = np.array([0., 1., 1.25])
    fx = np.stack([all_features[qid] for qid in split["train"]])
    fval = np.stack([all_features[qid] for qid in split["validation"]])
    train_quality = np.array([[score(qid, all_rankings[qid][method], benchmark.train_qrels)["ndcg@10"]
                               for method in ROUTES] for qid in split["train"]])
    val_quality = np.array([[score(qid, all_rankings[qid][method], benchmark.train_qrels)["ndcg@10"]
                             for method in ROUTES] for qid in split["validation"]])
    policies, candidate_models = [], []
    objective_lambda = .04
    for penalty in [0., .025, .05, .1, .2]:
        labels = (train_quality - penalty * costs).argmax(axis=1)
        router = fit_router(fx, labels, seed)
        routes = router.predict(fval)
        quality = float(val_quality[np.arange(len(routes)), routes].mean())
        cost = float(costs[routes].mean())
        policies.append({"training_cost_penalty": penalty, "validation_ndcg@10": quality,
                         "validation_proxy_cost": cost, "validation_objective": quality-objective_lambda*cost,
                         "train_route_labels": dict(Counter(ROUTES[i] for i in labels))})
        candidate_models.append(router)
    selected_policy = max(range(len(policies)), key=lambda i: policies[i]["validation_objective"])
    router = candidate_models[selected_policy]
    save_head(router, Path("checkpoints/router.safetensors"))
    config = {"seed": seed, "encoder_max_length": 192, "rrf_constant": 60, "candidate_depth": 100,
              "rerank_alpha": alpha, "routes": ROUTES, "proxy_costs": costs.tolist(),
              "routing_objective_lambda": objective_lambda, "selected_policy": policies[selected_policy],
              "adapter_sha256": adapter_hash}
    Path("checkpoints/config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Selected routing policy: {config['selected_policy']}", flush=True)

    # Selection ends here. Official BEIR test qrels are used only to report the fixed pipeline.
    details, route_counts = [], Counter()
    for qid in split["test"]:
        route = ROUTES[int(router.predict(all_features[qid])[0])]
        route_counts[route] += 1
        all_rankings[qid]["adaptive"] = all_rankings[qid][route]
        record = {"query_id": qid, "route": route, "metrics": {}, "top10": {}}
        for method in METHODS:
            ranking = all_rankings[qid][method]
            record["metrics"][method] = score(qid, ranking, benchmark.test_qrels)
            record["top10"][method] = [ids[i] for i in ranking[:10]]
        details.append(record)
    metrics = {method: aggregate([row["metrics"][method] for row in details]) for method in METHODS}

    # Warm, serial wall-clock timing; includes query tokenization/encoding and actual conditional dispatch.
    chosen = np.random.default_rng(seed).choice(split["test"], size=min(latency_queries, len(split["test"])), replace=False)
    latency = {method: [] for method in ["bm25", "hybrid_lora", "neural_rerank", "adaptive"]}

    def online(query: str, method: str) -> tuple[np.ndarray, str]:
        sp = bm25.scores(query)
        br = top_indices(sp, 100)
        route = method
        if method == "adaptive":
            route = ROUTES[int(router.predict(routing_features(query, sp, bm25))[0])]
        if route == "bm25":
            return br, route
        qv = encoder.encode([query])[0]
        ds = lora_docs @ qv
        lr = top_indices(ds, 100)
        fused = reciprocal_rank_fusion([br, lr], len(ids))
        candidates = top_indices(fused, 100)
        if route == "neural_rerank":
            terms = set(tokenize(query))
            cov = np.array([len(terms & bm25.text_tokens[i]) / max(len(terms), 1) for i in candidates])
            values = ranker.predict(pair_features(qv, lora_docs[candidates], sp[candidates], ds[candidates], cov))
            blended = (1-alpha) * normalized_scores(fused[candidates]) + alpha * values
            candidates = candidates[top_indices(blended, 100)]
        return candidates, route

    for method in latency:
        online(benchmark.queries[split["test"][0]], method)
    for index, qid in enumerate(chosen):
        # Rotate order to reduce systematic cache/CPU order effects.
        modes = list(latency)
        modes = modes[index % len(modes):] + modes[:index % len(modes)]
        for method in modes:
            t0 = time.perf_counter()
            online(benchmark.queries[qid], method)
            latency[method].append((time.perf_counter()-t0)*1000)
    latency_report = {method: {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                              "mean_ms": float(np.mean(values)), "queries": len(values)} for method, values in latency.items()}
    comparisons = {}
    for candidate, baseline in [("hybrid_lora", "hybrid_frozen"), ("dense_lora", "dense_frozen"),
                                ("adaptive", "bm25"), ("adaptive", "neural_rerank")]:
        a = np.array([row["metrics"][candidate]["ndcg@10"] for row in details])
        b = np.array([row["metrics"][baseline]["ndcg@10"] for row in details])
        comparisons[f"{candidate}-vs-{baseline}"] = paired_bootstrap(a, b, seed)
    report = {"dataset": "BEIR SciFact", "corpus": len(ids), "official_train": len(benchmark.train_qrels),
              "train": len(split["train"]), "validation": len(split["validation"]), "test": len(split["test"]),
              "seed": seed, "metrics": metrics, "latency": latency_report, "route_counts": dict(route_counts),
              "dense_call_fraction": sum(route_counts[m] for m in ROUTES[1:]) / len(details),
              "paired_ndcg_comparisons": comparisons, "reranker": ranker_report,
              "alpha_selection": alphas, "routing_selection": policies, "config": config,
              "timing_protocol": "40 seeded test queries; warm serial CPU; includes query encoding; cached corpus embeddings; excludes startup, indexing and generation",
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
                              "cpu_threads": torch.get_num_threads()},
              "runtime_seconds": time.perf_counter()-start,
              "limitations": ["one seed and one benchmark; pretrained contamination is unknown", "English biomedical retrieval does not establish Chinese enterprise generalization",
                              "BEIR test uses annotated SciFact validation, not the hidden SciFact leaderboard test", "LoRA adapts MiniLM, not Qwen; reranker is pooled-feature MLP, not a cross-encoder",
                              "routing costs are proxies; use reported measured latency for operational conclusions"]}
    Path("reports/retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("reports/per_query.jsonl").write_text("\n".join(json.dumps(row) for row in details)+"\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "latency": latency_report, "routes": dict(route_counts),
                      "seconds": report["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-queries", type=int, default=40)
    args = parser.parse_args()
    run(args.seed, args.latency_queries)
