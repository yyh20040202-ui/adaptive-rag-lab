"""Fixed actual BEIR test claims; citation acceptance is not QA correctness."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from .data import load_benchmark
from .engine import RetrievalEngine
from .resources import ResourceRecorder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="cross_encoder", help="Retrieval mode exposed by the engine")
    parser.add_argument("--output", default="reports/real_corpus_generation_v2.json")
    parser.add_argument("--evidence-budget", type=int, default=1400)
    args = parser.parse_args()
    data = load_benchmark()
    engine = RetrievalEngine.from_corpus("scifact")
    recorder = ResourceRecorder()
    keys = sorted(data.test_qrels)
    selected = [keys[i] for i in [0, 50, 100, 150, 200, 250]]
    records, started = [], time.perf_counter()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for qid in selected:
        recorder.sample()
        claim = data.queries[qid]
        result = engine.ask(claim, args.mode, evidence_budget=args.evidence_budget)
        packed_ids = {item["source_id"] for item in result.get("evidence", [])}
        records.append({"query_id": qid, "claim": claim,
                        "retrieval_qrels": data.test_qrels[qid],
                        "relevant_source_in_packed_context": bool(packed_ids & data.test_qrels[qid].keys()),
                        "result": result})
        progress = {"status": "running", "planned_cases": len(selected), "cases": len(records),
                    "answered": sum(not row["result"]["abstain"] for row in records),
                    "fixed_query_ids": selected, "results": records}
        output.write_text(json.dumps(progress,indent=2,ensure_ascii=False),encoding="utf-8")
        print(f"SciFact {qid}: abstain={result['abstain']}; citations={len(result['citations'])}", flush=True)
    failures = Counter(reason for record in records for reason in record["result"].get("audit_failures", []))
    report = {"status": "complete", "corpus": "real BEIR SciFact abstracts", "cases": len(records),
              "retrieval_mode": args.mode, "evidence_budget": args.evidence_budget,
              "fixed_query_ids": selected, "elapsed_seconds": time.perf_counter()-started,
              "answered": sum(not row["result"]["abstain"] for row in records),
              "relevant_source_in_packed_context": sum(row["relevant_source_in_packed_context"] for row in records),
              "audit_failure_counts": dict(failures),
              "resources": recorder.result(),
              "retrieval_config": engine.quality_config if args.mode == "cross_encoder" else engine.config,
              "description": "Same fixed six claims as the original run; no QA correctness labels. Answered means the output passed mechanical source/quote provenance checks only. Qrels indicate retrieval relevance. Span selection constrains quotes, so quote validity is not an independently measured semantic quality score.",
              "results": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
