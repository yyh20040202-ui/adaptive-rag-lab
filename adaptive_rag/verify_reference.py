"""Check all committed benchmark rankings against the actual inference engine."""
import json
from pathlib import Path
from .data import load_benchmark
from .engine import RetrievalEngine


if __name__ == "__main__":
    data = load_benchmark()
    engine = RetrievalEngine.from_corpus("scifact")
    records = [json.loads(line) for line in Path("reports/per_query.jsonl").read_text().splitlines()]
    checks, failures = 0, []
    for row in records:
        for method in ["bm25", "hybrid_lora", "neural_rerank", "adaptive"]:
            result = engine.search(data.queries[row["query_id"]], method, 10)
            actual = [item["source_id"] for item in result["hits"]]
            if actual != row["top10"][method]:
                failures.append({"query_id": row["query_id"], "method": method,
                                 "actual": actual, "expected": row["top10"][method]})
            if method == "adaptive" and result["route"] != row["route"]:
                failures.append({"query_id": row["query_id"], "method": "route_mismatch"})
            checks += 1
    report = {"queries": len(records), "ranking_checks": checks, "failures": failures,
              "passed": not failures, "scope": "all official test top-10 rankings in four served modes; includes live query encoding"}
    Path("reports/reference_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)
