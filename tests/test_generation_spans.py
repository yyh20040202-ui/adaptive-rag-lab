import json

import httpx
import pytest

from adaptive_rag.data import Document
from adaptive_rag.evidence import Evidence, best_excerpt, citation_catalog, evidence_json, pack_evidence, validate_answer
from adaptive_rag.generator import LocalGenerator, resolve_span_payload
from adaptive_rag.answerability import missing_or_echo_answer


def test_body_spans_preserve_unicode_whitespace_and_absolute_offsets():
    content = "Research title\nFirst finding: CD4+ T cells increase chemokines.\nSecond finding uses IFN-γ."
    evidence = Evidence("paper", "Research title", content, 0, len(content))
    catalog, lookup = citation_catalog([evidence])
    assert catalog[0]["title"] == "Research title"
    assert all(span["quote"] != "Research title" for span in lookup.values())
    assert any("IFN-γ" in span["quote"] for span in lookup.values())
    for span in lookup.values():
        assert content[span["start"]:span["end"]] == span["quote"]
    span_id = next(iter(lookup))
    decoded = resolve_span_payload({"abstain": False, "answer": "The study reports a finding.",
                                   "citations": [{"span_id": span_id}]}, lookup)
    audited = validate_answer(decoded, [evidence])
    assert not audited["abstain"]
    assert audited["citations"][0]["quote"] == lookup[span_id]["quote"]


def test_long_scientific_sentence_has_bounded_exact_candidates():
    text = "The study found " + "important cellular mechanisms and quantitative outcomes; "*20 + "a final outcome."
    evidence = Evidence("paper", "Study", text, 175, 175+len(text))
    _, lookup = citation_catalog([evidence])
    assert len(lookup) > 1
    assert all(8 <= len(span["quote"]) <= 360 for span in lookup.values())
    for span in lookup.values():
        assert text[span["start"]-175:span["end"]-175] == span["quote"]
    assert any("final outcome" in span["quote"] for span in lookup.values())


def test_title_alone_does_not_become_citable_evidence():
    item = Evidence("paper", "A descriptive research title", "A descriptive research title\n", 0, 29)
    assert citation_catalog([item]) == ([], {})
    item = Evidence("paper", "A descriptive research title", "A descriptive research title", 0, 28)
    assert citation_catalog([item]) == ([], {})
    long_title = "A very long research title "*10
    text = long_title+"\nThe paper reports a specific cellular finding."
    item = Evidence("paper", long_title[:180], text, 0, len(text))
    _, lookup = citation_catalog([item])
    assert len(lookup) == 1
    assert next(iter(lookup.values()))["quote"] == "The paper reports a specific cellular finding."


@pytest.mark.parametrize("citations", [[{"span_id": "fabricated"}],
                                     [{"span_id": "s1_1"}, {"span_id": "s1_1"}],
                                     [{"span_id": "s1_1", "quote": "invented"}]])
def test_unknown_duplicate_or_extra_citation_fields_fail_closed(citations):
    with pytest.raises(ValueError):
        resolve_span_payload({"abstain": False, "answer": "A finding.", "citations": citations},
                             {"s1_1": {"source_id": "doc", "quote": "Original quote."}})


def test_budget_is_for_actual_model_catalog_and_keeps_later_source():
    documents = [Document(str(i), f"Paper {i}", "Background facts. "*35 +
                          f"The unique mechanism{i} increases the measured outcome. " + "Further context. "*30)
                 for i in range(4)]
    counter = lambda value: (len(value.encode("utf-8"))+3)//4
    evidence, tokens = pack_evidence(documents, "mechanism3 measured outcome", 1400, counter)
    assert tokens == counter(evidence_json(evidence)) <= 1400
    assert len(evidence) == 4
    assert "mechanism3" in evidence[-1].text
    assert tokens > 1100  # the original 900-character cap left much of this budget unused
    for item, document in zip(evidence, documents):
        assert document.content[item.start:item.end] == item.text


def test_clipping_preserves_query_term_late_in_long_sentence():
    document = Document("doc", "Study", "Unrelated background "*90 + "special_marker decisive outcome " + "extra material "*30)
    excerpt = best_excerpt(document, "special_marker decisive outcome", 220)
    assert "special_marker" in excerpt.text
    assert document.content[excerpt.start:excerpt.end] == excerpt.text


def test_generator_uses_enum_spans_and_reads_runtime_model_label(monkeypatch):
    monkeypatch.setenv("LLAMA_MODEL_LABEL", "Qwen3-1.7B Q4_K_M")
    captured = {}

    def respond(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "abstain": False, "answer": "The study reports increased chemokines and early viral control.",
            "citations": [{"span_id": "s1_1"}]}),}, "finish_reason": "stop"}], "usage": {}})

    generator = LocalGenerator()
    generator.client = httpx.Client(base_url="http://127.0.0.1:8087", transport=httpx.MockTransport(respond))
    text = "CD4+ T cells increase chemokines and early viral control."
    evidence = [Evidence("paper", "Study", text, 0, len(text))]
    payload, stats = generator.generate("Early chemokines improve viral control.", evidence)
    assert payload["citations"] == [{"source_id": "paper", "quote": text}]
    assert "Qwen3-1.7B" in stats["model"]
    assert not validate_answer(payload, evidence)["abstain"]
    schema = captured["response_format"]["schema"]
    enum = schema["oneOf"][0]["properties"]["citations"]["items"]["properties"]["span_id"]["enum"]
    assert enum == ["s1_1"]
    assert "declarative claim" in captured["messages"][0]["content"]


def test_labelled_claim_echo_is_rejected_but_explanation_is_kept():
    claim = "TET2 loss contributes to myeloid transformation."
    assert missing_or_echo_answer(claim, "Supported: "+claim) == "question_echo"
    assert missing_or_echo_answer(claim, "The evidence supports the claim: "+claim) == "question_echo"
    assert missing_or_echo_answer(claim, "Supported in the reported setting: "+claim) == "question_echo"
    assert missing_or_echo_answer(claim, "Supported: the knockout study reports myeloid transformation following TET2 loss.") is None
