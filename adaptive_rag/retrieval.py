"""Sparse retrieval, deterministic fusion, and retrieval-derived features."""
from __future__ import annotations

import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Score descending, corpus position ascending: reproducible even on ties."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector")
    return np.lexsort((np.arange(len(scores)), -scores))[:max(0, min(k, len(scores)))]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = .75):
        if not texts:
            raise ValueError("An index needs documents")
        self.vectorizer = CountVectorizer(analyzer=tokenize, dtype=np.float32)
        counts = self.vectorizer.fit_transform(texts).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(texts) - document_frequency + .5) / (document_frequency + .5))
        norm = k1 * (1 - b + b * lengths / max(float(lengths.mean()), 1))
        rows = np.repeat(np.arange(len(texts)), np.diff(counts.indptr))
        weights = counts.copy()
        weights.data = (counts.data * (k1 + 1) / (counts.data + norm[rows])) * idf[counts.indices]
        self.weights = weights.tocsr()
        self.text_tokens = [set(tokenize(text)) for text in texts]

    def scores(self, query: str) -> np.ndarray:
        terms = self.vectorizer.transform([query])
        terms.data[:] = 1  # repeated query terms do not multiply evidence
        return np.asarray((self.weights @ terms.T).toarray()).ravel()

    def search(self, query: str, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        scores = self.scores(query)
        indices = top_indices(scores, k)
        return indices, scores[indices]


def reciprocal_rank_fusion(rankings: list[np.ndarray], count: int,
                           constant: float = 60, depth: int = 100) -> np.ndarray:
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    # float64 prevents asymmetric rounding of equal two-way rank sums.
    scores = np.zeros(count, dtype=np.float64)
    for ranking in rankings:
        ranking = np.asarray(ranking[:depth], dtype=int)
        if len(np.unique(ranking)) != len(ranking) or ((ranking < 0) | (ranking >= count)).any():
            raise ValueError("Invalid ranking")
        scores[ranking] += 1 / (constant + np.arange(1, len(ranking) + 1))
    return scores


def normalized_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    span = float(np.ptp(values))
    return (values - values.min()) / span if span > 1e-9 else np.zeros_like(values)


def pair_features(query: np.ndarray, documents: np.ndarray, bm25: np.ndarray,
                  dense: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """A neural learning-to-rank model, explicitly not a token cross-encoder."""
    q = np.broadcast_to(query, documents.shape)
    scalar = np.column_stack([np.log1p(np.maximum(bm25, 0)) / 5, dense, coverage])
    return np.concatenate([q * documents, np.abs(q - documents), scalar], axis=1).astype(np.float32)


def routing_features(query: str, scores: np.ndarray, bm25: BM25) -> np.ndarray:
    order = top_indices(scores, min(10, len(scores)))
    top = scores[order]
    tokens = set(tokenize(query))
    maximum = float(top[0]) if len(top) else 0
    gap = float(top[0] - top[1]) / max(maximum, 1e-8) if len(top) > 1 else 0
    p = np.exp((top - maximum) / max(float(np.std(top)), 1))
    p /= max(float(p.sum()), 1e-8)
    entropy = float(-(p * np.log(p + 1e-8)).sum() / max(np.log(max(len(top), 2)), 1))
    coverage = len(tokens & bm25.text_tokens[int(order[0])]) / max(len(tokens), 1) if len(order) else 0
    return np.array([np.log1p(maximum), gap, entropy, coverage,
                     np.log1p(len(tokens)), float((scores > 0).mean()),
                     float(not tokens)], dtype=np.float32)
