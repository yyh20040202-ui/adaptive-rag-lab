from adaptive_rag.data import Document
from adaptive_rag.evidence import Evidence, pack_evidence, validate_answer, evidence_json
import pytest


def payload(sid="d", quote="Hotel limit is CNY 500."):
    return {"answer": "The limit is CNY 500.", "abstain": False,
            "citations": [{"source_id": sid, "quote": quote}]}


def test_exact_quote_maps_back_to_document_offsets():
    evidence = Evidence("d", "Policy", "Hotel limit is CNY 500. Submit a receipt.", 20, 61)
    result = validate_answer(payload(), [evidence])
    assert not result["abstain"]
    assert result["citations"][0]["start"] == 20
    assert result["citations"][0]["end"] == 43


def test_fabricated_source_and_quote_are_blocked():
    evidence = [Evidence("d", "Policy", "Hotel limit is CNY 500.", 0, 22)]
    assert validate_answer(payload(sid="invented"), evidence)["audit_failures"] == ["unknown_source"]
    assert validate_answer(payload(quote="Hotel limit is CNY 900."), evidence)["audit_failures"] == ["quote_not_in_context"]
    assert validate_answer({"answer": "No evidence.", "abstain": False, "citations": []}, evidence)["abstain"]


def test_budget_includes_serialization_and_clipped_offsets():
    doc = Document("d", "Budget policy", "hotel budget "*200)
    # Byte counter is deliberately stricter than the model's tokenizer.
    evidence, count = pack_evidence([doc, doc], "hotel", 300, lambda s: len(s.encode("utf-8")))
    assert len(evidence) == 1
    assert count <= 300
    chunk = evidence[0]
    assert doc.content[chunk.start:chunk.end] == chunk.text


def test_correct_quote_does_not_certify_semantic_truth():
    evidence = [Evidence("d", "Policy", "Hotel limit is CNY 500.", 0, 22)]
    data = payload()
    data["answer"] = "The hotel limit is 900."
    assert not validate_answer(data, evidence)["abstain"]  # explicit, tested limitation


def test_preferred_late_span_keeps_quote_offsets_budget_and_expansion_bounds():
    quote = "Quokka reimbursement is CNY 321."
    source = Document("d", "Travel policy", "Filing reimbursement instructions. " * 180 + quote + " Preserve receipts.")
    start = source.content.index(quote)
    end = len(source.content)
    evidence, count = pack_evidence([source], "quokka reimbursement", 400,
                                    lambda value: len(value.encode("utf-8")),
                                    preferred_spans={source.id: {"start": start, "end": end}})
    assert len(evidence) == 1 and count <= 400
    assert count == len(evidence_json(evidence).encode("utf-8"))
    selected = evidence[0]
    assert selected.start >= start and selected.end <= end
    assert source.content[selected.start:selected.end] == selected.text
    assert quote in selected.text
    result = validate_answer(payload(quote=quote), evidence)
    assert not result["abstain"]
    citation = result["citations"][0]
    assert citation["start"] == start
    assert source.content[citation["start"]:citation["end"]] == quote


@pytest.mark.parametrize("span", [{"start": -1, "end": 20}, {"start": 8, "end": 8},
                                  {"start": 0, "end": 9999}, {"start": True, "end": 20}, {}])
def test_preferred_span_rejects_invalid_original_offsets(span):
    source = Document("d", "Policy", "Hotel limit is CNY 500.")
    with pytest.raises(ValueError, match="Preferred evidence span"):
        pack_evidence([source], "hotel", 300, len, preferred_spans={"d": span})


def test_no_preferred_span_preserves_default_packing():
    source = Document("d", "Policy", "Hotel limit is CNY 500.")
    assert pack_evidence([source], "hotel", 300, len) == pack_evidence([source], "hotel", 300, len, preferred_spans={})
