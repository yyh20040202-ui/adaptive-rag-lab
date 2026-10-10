"""Offline concurrency regression, not a generation-quality benchmark."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from adaptive_rag.data import Document
from adaptive_rag.engine import RetrievalEngine


def test_generation_requests_across_corpora_do_not_overlap():
    state = {"active": 0, "peak": 0}
    state_lock = threading.Lock()

    class FixtureGenerator:
        def healthy(self):
            return True

        def count_tokens(self, text):
            return len(text.split())

        def generate(self, query, evidence):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(.04)
            with state_lock:
                state["active"] -= 1
            return {"abstain": False, "answer": "The daily ceiling is CNY 321.",
                    "citations": [{"source_id": evidence[0].source_id,
                                   "quote": "The hotel limit is CNY 321 per day."}]}, {}

    engines = [RetrievalEngine([Document(str(i), "Travel", "The hotel limit is CNY 321 per day.")],
                               domain="demo", generator=FixtureGenerator()) for i in range(2)]
    start = threading.Barrier(2)

    def request(engine):
        start.wait(timeout=3)
        return engine.ask("hotel limit", "bm25")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, engines))
    assert state["peak"] == 1
    assert all(not result["abstain"] and result["citations"] for result in results)
