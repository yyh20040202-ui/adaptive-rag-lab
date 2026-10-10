"""Evidence packing and fail-closed source/quote checks, without an LLM judge."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable, Mapping

from .data import Document
from .retrieval import tokenize


@dataclass(frozen=True)
class Evidence:
    source_id: str
    title: str
    text: str
    start: int
    end: int


def citation_catalog(evidence: list[Evidence]) -> tuple[list[dict], dict[str, dict]]:
    """Expose bounded, exact source spans once, rather than asking an LLM to copy them.

    A span ID is an output constraint, not a relevance or entailment label. Titles
    are metadata and cannot be selected as proof. Every selectable quote remains
    a contiguous slice of the packed source text and is audited again downstream.
    """
    catalog, lookup = [], {}
    for source_number, item in enumerate(evidence, 1):
        text = item.text
        offset = 0
        if item.start == 0 and item.title:
            if item.title.startswith(text.strip()):
                continue  # clipped or complete title without an abstract
            if text.startswith(item.title):
                newline = text.find("\n")
                if newline < 0:
                    continue
                offset = newline+1  # also handles a metadata title capped at 180 characters
        spans = []
        # Split only on sentence-final punctuation followed by whitespace. This
        # preserves decimals, gene names and punctuation in scientific abstracts.
        boundaries = [offset]
        boundaries.extend(match.end() for match in re.finditer(r"(?<=[.!?。！？])\s+|\n+", text[offset:]))
        # The regular expression above is evaluated on the suffix.
        boundaries = [offset] + [offset+b for b in boundaries[1:]] + [len(text)]
        pending_start = offset
        for end in sorted(set(boundaries[1:])):
            if len(text[pending_start:end].strip()) < 8:
                continue
            start = pending_start
            pending_start = end
            while start < end:
                chunk_end = min(end, start+360)
                if chunk_end < end:
                    # Prefer a clause/word boundary; leave at least 8 characters
                    # for the final piece instead of silently discarding a fact.
                    boundary = max(text.rfind(mark, start+100, chunk_end) for mark in (";", ",", "，", "；", " "))
                    if boundary >= start+100:
                        chunk_end = boundary+1
                    chunk_end = min(chunk_end, end-8)
                raw = text[start:chunk_end]
                quote = raw.strip()
                local_start = start+len(raw)-len(raw.lstrip())
                if 8 <= len(quote) <= 360:
                    span_id = f"s{source_number}_{len(spans)+1}"
                    spans.append({"span_id": span_id, "text": quote})
                    lookup[span_id] = {"source_id": item.source_id, "quote": quote,
                                       "start": item.start+local_start,
                                       "end": item.start+local_start+len(quote)}
                start = chunk_end
        if spans:
            catalog.append({"source_id": item.source_id, "title": item.title, "spans": spans})
    return catalog, lookup


def evidence_json(evidence: list[Evidence]) -> str:
    """The actual citation catalog sent to the model, including JSON overhead."""
    return json.dumps(citation_catalog(evidence)[0], ensure_ascii=False, separators=(",", ":"))


def best_excerpt(document: Document, query: str, max_chars: int = 1400,
                 source_span: tuple[int, int] | None = None) -> Evidence:
    content = document.content
    origin = 0
    if source_span is not None:
        origin, boundary = source_span
        if type(origin) is not int or type(boundary) is not int or not 0 <= origin < boundary <= len(content):
            raise ValueError("Preferred evidence span must index the original document content")
        content = content[origin:boundary]
    if len(content) <= max_chars:
        return Evidence(document.id, document.title[:180], content, origin, origin+len(content))
    terms = set(tokenize(query))
    # Offsets always index document.content, including the title and its newline.
    spans = []
    for match in re.finditer(r"[^.!?。！？\n]+[.!?。！？]?", content):
        if origin == 0 and match.end() <= len(document.title):
            continue  # titles are not sufficient evidence of a research finding
        words = set(tokenize(match.group()))
        score = len(terms & words) / (len(words)+8)**.5
        spans.append((score, match.start(), match.end()))
    if spans:
        _, start, end = max(spans, key=lambda row: (row[0], -row[1]))
        if end-start < max_chars:
            previous = [s for _, s, e in spans if e <= start and start-s <= min(200, max_chars-(end-start))]
            start = min(previous) if previous else start
            end = min(len(content), start+max_chars)
        else:
            # Do not lose a query term late in a long sentence when clipping.
            matches = [m for m in re.finditer(r"\b[\w-]+\b", content[start:end]) if m.group().lower() in terms]
            if matches:
                anchor = start+matches[len(matches)//2].start()
                start = max(start, anchor-max_chars//2)
            end = min(end, start+max_chars)
    else:
        start, end = 0, min(len(content), max_chars)
    return Evidence(document.id, document.title[:180], content[start:end], origin+start, origin+end)


def pack_evidence(documents: list[Document], query: str, budget: int,
                  count_tokens: Callable[[str], int], max_sources: int = 4,
                  preferred_spans: Mapping[str, Mapping[str, int]] | None = None) -> tuple[list[Evidence], int]:
    if budget < 64:
        raise ValueError("Evidence budget must be at least 64 tokens")
    if max_sources < 1:
        raise ValueError("max_sources must be positive")
    selected, used_ids = [], set()
    source_cap = min(max_sources, max(1, budget//160))
    for document in documents:
        if document.id in used_ids or not document.content:
            continue
        selected.append(document)
        used_ids.add(document.id)
        if len(selected) == source_cap:
            break
    spans = {}
    for document in selected:
        if preferred_spans is None or document.id not in preferred_spans:
            continue
        span = preferred_spans[document.id]
        if not isinstance(span, Mapping):
            raise ValueError("Preferred evidence span requires start and end offsets")
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document.content):
            raise ValueError("Preferred evidence span must index the original document content")
        spans[document.id] = (start, end)
    accepted, limits, included = [], [], []
    # Reserve a fair share for later retrieved sources. A relevant paper in slot
    # four must not lose its evidence merely because slot one has a long abstract.
    for number, document in enumerate(selected):
        spent = count_tokens(evidence_json(accepted))
        allowance = max(1, (budget-spent)//(len(selected)-number))
        span = spans.get(document.id)
        available_chars = span[1]-span[0] if span is not None else len(document.content)
        chars = min(available_chars, max(32, allowance*3))
        while chars >= 16:
            excerpt = best_excerpt(document, query, chars, source_span=span)
            candidate = accepted+[excerpt]
            total = count_tokens(evidence_json(candidate))
            if citation_catalog([excerpt])[0] and total <= spent+allowance and total <= budget:
                accepted = candidate
                included.append(document)
                limits.append(chars)
                break
            chars -= max(8, chars//5)
    # Spend remaining budget on additional surrounding source text, checking the
    # complete catalog after every expansion with the real model tokenizer.
    changed = True
    while changed:
        changed = False
        for number, document in enumerate(included):
            span = spans.get(document.id)
            ceiling = min(span[1]-span[0] if span is not None else len(document.content), 3200)
            if limits[number] >= ceiling:
                continue
            larger = min(ceiling, limits[number]+160)
            candidate = accepted.copy()
            candidate[number] = best_excerpt(document, query, larger, source_span=span)
            if count_tokens(evidence_json(candidate)) <= budget:
                accepted, limits[number], changed = candidate, larger, True
    count = count_tokens(evidence_json(accepted))
    if count > budget:
        raise ValueError("Evidence serialization overhead exceeds the token budget")
    return accepted, count


def validate_answer(payload: dict, evidence: list[Evidence]) -> dict:
    """Exact quotes prove provenance, not whether the answer follows from the quote."""
    sources = {item.source_id: item for item in evidence}
    failures, verified = [], []
    if not isinstance(payload, dict):
        return abstention(["invalid_payload"])
    if type(payload.get("abstain")) is not bool:
        failures.append("invalid_abstain_flag")
    if payload.get("abstain") is True:
        return abstention(["model_abstained"])
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 1600:
        failures.append("invalid_answer")
    citations = payload.get("citations")
    if not isinstance(citations, list) or not 1 <= len(citations) <= 4:
        failures.append("missing_or_invalid_citations")
        citations = []
    seen = set()
    for citation in citations:
        if not isinstance(citation, dict):
            failures.append("invalid_citation")
            continue
        sid, quote = citation.get("source_id"), citation.get("quote")
        if not isinstance(sid, str) or sid not in sources:
            failures.append("unknown_source")
            continue
        if not isinstance(quote, str) or not 8 <= len(quote) <= 400:
            failures.append("invalid_quote_length")
            continue
        local_start = sources[sid].text.find(quote)
        if local_start < 0:
            failures.append("quote_not_in_context")
            continue
        if (sid, quote) in seen:
            failures.append("duplicate_citation")
            continue
        seen.add((sid, quote))
        verified.append({"source_id": sid, "title": sources[sid].title, "quote": quote,
                         "start": sources[sid].start+local_start,
                         "end": sources[sid].start+local_start+len(quote), "quote_verified": True})
    if failures:
        return abstention(failures)
    return {"answer": answer.strip(), "abstain": False, "citations": verified, "audit_failures": []}


def abstention(reasons: list[str]) -> dict:
    return {"answer": "现有资料不足以提供经过引用核验的回答。请补充资料或调整问题。", "abstain": True,
            "citations": [], "audit_failures": reasons}
