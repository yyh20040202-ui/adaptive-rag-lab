import math
import numpy as np
import pytest
from adaptive_rag.metrics import query_metrics, paired_bootstrap
from adaptive_rag.retrieval import BM25, reciprocal_rank_fusion, routing_features, top_indices


def test_sparse_retrieval_handles_unknown_terms_and_ties():
    index = BM25(["hotel allowance Beijing", "data privacy policy", "hotel allowance Shanghai"])
    assert index.search("privacy", 1)[0].tolist() == [1]
    assert index.search("zznotinvocabulary", 3)[0].tolist() == [0, 1, 2]
    assert index.scores("privacy privacy").tolist() == index.scores("privacy").tolist()
    assert np.isfinite(routing_features("", np.zeros(3), index)).all()


def test_rrf_agreement_and_rank_positions():
    fused = reciprocal_rank_fusion([np.array([2, 0, 1]), np.array([2, 1, 0])], 3)
    assert top_indices(fused, 3).tolist() == [2, 0, 1]
    assert fused[2] == pytest.approx(2/61)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([np.array([0, 0])], 2)


def test_hand_checked_graded_metrics_at_cutoff():
    score = query_metrics(["irrelevant", "high", "low"], {"high": 2, "low": 1}, k=2)
    assert score["mrr@2"] == .5
    assert score["recall@2"] == .5
    assert score["recall@5"] == 1.
    assert score["ndcg@2"] == pytest.approx((3/math.log2(3))/(3+1/math.log2(3)))
    with pytest.raises(ValueError):
        query_metrics(["high", "high"], {"high": 1})


def test_paired_bootstrap_preserves_pairing():
    result = paired_bootstrap(np.array([.4, .6, .8]), np.array([.3, .5, .7]), iterations=100)
    assert result["delta"] == pytest.approx(.1)
    assert result["ci95"] == pytest.approx([.1, .1])
