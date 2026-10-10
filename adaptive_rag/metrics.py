"""Transparent graded retrieval metrics and paired bootstrap uncertainty."""
from __future__ import annotations
import numpy as np


def query_metrics(ranking: list[str], relevant: dict[str, int], k: int = 10) -> dict:
    if not relevant or k <= 0:
        raise ValueError("Positive qrels and cutoff are required")
    if len(set(ranking)) != len(ranking):
        raise ValueError("A ranking cannot contain duplicates")
    all_hits = [float(relevant.get(did, 0)) for did in ranking[:max(k, 5)]]
    hits = all_hits[:k]
    discounts = np.log2(np.arange(2, len(hits) + 2))
    dcg = float(((np.power(2, hits) - 1) / discounts).sum())
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = float(((np.power(2, ideal) - 1) / np.log2(np.arange(2, len(ideal) + 2))).sum())
    first = next((i + 1 for i, h in enumerate(hits) if h > 0), None)
    return {f"ndcg@{k}": dcg / idcg if idcg else 0,
            f"mrr@{k}": 1 / first if first else 0,
            f"recall@{k}": sum(h > 0 for h in hits) / len(relevant),
            "recall@5": sum(h > 0 for h in all_hits[:5]) / len(relevant)}


def aggregate(per_query: list[dict]) -> dict:
    if not per_query:
        raise ValueError("Cannot aggregate an empty benchmark")
    return {key: float(np.mean([row[key] for row in per_query])) for key in per_query[0]}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = 42,
                     iterations: int = 2000) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Paired measurements must have the same non-empty shape")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(iterations, len(a)))
    delta = (a - b)[indices].mean(axis=1)
    return {"delta": float((a - b).mean()), "ci95": [float(v) for v in np.quantile(delta, [.025, .975])],
            "method": "paired query bootstrap", "iterations": iterations, "seed": seed}
