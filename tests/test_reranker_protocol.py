import json
import numpy as np
import pytest
import httpx
from adaptive_rag.reranker_client import GGUFReranker


class Recorder:
    def sample(self):
        return {}


def make_client(handler):
    client = GGUFReranker.__new__(GGUFReranker)
    client.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8089")
    client.max_length = 384
    client.batch_size = 8
    client.recorder = Recorder()
    client.tokenize = lambda text: tuple(range(len(text.split())))
    client.document_prefix = lambda text, limit: text
    return client


def test_response_rank_order_is_restored_to_original_pair_indices():
    def handler(request):
        body = json.loads(request.content)
        assert body["documents"] == ["first paper", "second paper", "third paper"]
        return httpx.Response(200, json={"results": [{"index": 2, "relevance_score": .9}, {"index": 0, "relevance_score": .6}, {"index": 1, "relevance_score": .1}]})
    model = make_client(handler)
    np.testing.assert_allclose(model.predict("a claim", ["first paper", "second paper", "third paper"]), [.6, .1, .9])


@pytest.mark.parametrize("rows", [[], [{"index": 0, "relevance_score": .8}, {"index": 0, "relevance_score": .2}], [{"index": 0, "relevance_score": float("inf")}, {"index": 1, "relevance_score": .4}]])
def test_incomplete_duplicate_or_nonfinite_pair_scores_fail_closed(rows):
    model = make_client(lambda request: httpx.Response(200, json={"results": rows}) if all(np.isfinite(row["relevance_score"]) for row in rows) else httpx.Response(200, content=b'{"results":[{"index":0,"relevance_score":1e999},{"index":1,"relevance_score":0.4}]}', headers={"Content-Type": "application/json"}))
    with pytest.raises(ValueError):
        model.predict("claim", ["first", "second"])


def test_overlong_query_stops_before_pair_requests():
    model = make_client(lambda request: pytest.fail("No inference request should be sent"))
    with pytest.raises(ValueError, match="128"):
        model.predict("word "*129, ["paper"])
