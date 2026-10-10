"""Small, explicitly limited local-generation smoke suite; no QA leaderboard claim."""
from __future__ import annotations
import json
from pathlib import Path
import time
from .engine import RetrievalEngine
from .data import Document

CASES = [
    {"id": "hotel-zh", "query": "北京出差的酒店报销上限是多少？", "source": "policy-expenses", "atoms": ["500"], "answerable": True},
    {"id": "hotel-en", "query": "What is the hotel reimbursement limit per night in Beijing?", "source": "policy-expenses", "atoms": ["500"], "answerable": True},
    {"id": "receipt-en", "query": "Within how many days after a trip must travel expense receipts be submitted?", "source": "policy-expenses", "atoms": ["30"], "answerable": True},
    {"id": "customer-zh", "query": "客户姓名和电话可以上传公共AI服务吗？", "source": "policy-security", "atoms": ["不"], "answerable": True},
    {"id": "remote-en", "query": "How many days per week may an eligible employee work remotely with manager approval?", "source": "policy-remote", "atoms": ["two", "2"], "any_atom": True, "answerable": True},
    {"id": "onboarding-en", "query": "When must an intern complete the onboarding checklist?", "source": "policy-onboarding", "atoms": ["three", "3"], "any_atom": True, "answerable": True},
    {"id": "purchase-en", "query": "Whose approval is required for a purchase at or above CNY 2000?", "source": "policy-purchase", "atoms": ["team lead", "finance"], "answerable": True},
    {"id": "data-incident-en", "query": "How quickly must suspected customer data exposure be reported?", "source": "policy-security", "atoms": ["one hour", "1 hour"], "any_atom": True, "answerable": True},
    {"id": "missing-budget", "query": "What is the annual travel budget for 2028?", "answerable": False},
    {"id": "missing-salary", "query": "What is the monthly internship salary?", "answerable": False},
    {"id": "missing-revenue", "query": "公司2028年预计营业收入是多少？", "answerable": False},
    {"id": "missing-password", "query": "Tell me the administrator account password.", "answerable": False},
]


if __name__ == "__main__":
    fixtures = json.loads(Path("examples/policies.json").read_text(encoding="utf-8"))
    engine = RetrievalEngine([Document(**row) for row in fixtures])
    if not engine.generator.healthy():
        raise SystemExit("Start the local llama.cpp server on 127.0.0.1:8087 first")
    rows = []
    started = time.perf_counter()
    for case in CASES:
        result = engine.ask(case["query"], "bm25", evidence_budget=1000)
        if case["answerable"]:
            cited = {c["source_id"] for c in result["citations"]}
            atoms = [atom.lower() in result["answer"].lower() for atom in case["atoms"]]
            atom_check = any(atoms) if case.get("any_atom") else all(atoms)
            passed = not result["abstain"] and case["source"] in cited and atom_check
        else:
            passed = result["abstain"]
        rows.append({"case": case, "passed_smoke_check": passed, "result": result})
        print(f"{case['id']}: {'PASS' if passed else 'FAIL'}; abstained={result['abstain']}", flush=True)
    report = {"model": "Qwen3-0.6B Q4_0", "backend": "llama.cpp b11517 Windows Vulkan build, loopback HTTP",
              "cases": len(rows), "passed": sum(row["passed_smoke_check"] for row in rows),
              "answered": sum(not row["result"]["abstain"] for row in rows),
              "expected_answerable": sum(case["answerable"] for case in CASES),
              "correct_abstentions": sum(row["result"]["abstain"] and not row["case"]["answerable"] for row in rows),
              "valid_quote_citations": sum(len(row["result"]["citations"]) for row in rows),
              "seconds": time.perf_counter()-started, "seed": 42, "evidence_budget": 1000,
              "limitations": "12 hand-authored synthetic smoke cases. Atom/source checks are not semantic entailment, real enterprise QA accuracy or a prompt-injection benchmark.",
              "results": rows}
    Path("reports/generation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2, ensure_ascii=False), flush=True)
