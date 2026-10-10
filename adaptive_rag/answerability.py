"""Conservative heuristics for empty, unrelated, or explicitly missing evidence.

These guards are not semantic entailment classifiers. They intentionally trade
some recall for abstention and do not replace a domain-calibrated answerability model.
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

GENERIC_CHINESE = {"多少", "什么", "如何", "可以", "是否", "的是", "是多", "为什", "请问", "告诉", "我们"}
MISSING_FACT = re.compile(r"\bnot\s+(?:specified|mentioned|provided|available|included)\b|"
                          r"\b(?:cannot|can't)\s+(?:determine|answer)\b|"
                          r"(?:未|没有)(?:提供|说明|规定)|无法(?:确定|回答)|资料不足", re.IGNORECASE)


def informative_terms(text: str) -> set[str]:
    english = {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in ENGLISH_STOP_WORDS}
    chinese = set()
    for span in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        chinese.update(span[i:i+2] for i in range(len(span)-1))
    return english | (chinese - GENERIC_CHINESE)


def has_topical_overlap(query: str, evidence: list[str]) -> bool:
    terms = informative_terms(query)
    if not terms:
        return False
    text = "\n".join(evidence).lower()
    return any(term in text for term in terms)


def missing_query_anchor(query: str, evidence: list[str]) -> str | None:
    """Conservative literal guard for formal English identifiers, not entailment.

    Keep numeric compounds and acronyms distinguishable; a paper mentioning
    biomaterials alone cannot establish a statement about 0-dimensional ones.
    Synonym-only sources may be rejected and need a calibrated semantic model.
    """
    anchors = re.findall(r"\b(?:\d+-[A-Za-z][\w-]*|[A-Z]{2,}[A-Za-z0-9]*|[a-z]+[A-Z][A-Za-z0-9]*)\b", query)
    text = "\n".join(evidence)
    for anchor in anchors:
        pattern = r"\b"+re.escape(anchor)+(r"\d*\b" if anchor.isupper() else r"\b")
        if not re.search(pattern, text, re.IGNORECASE):
            return anchor
    return None


def missing_or_echo_answer(query: str, answer: str) -> str | None:
    normalized = lambda value: re.sub(r"[\W_]+", "", value.lower())
    # A support label followed only by the original claim does not explain a
    # source finding. Preserve substantive answers that also describe evidence.
    remainder = re.sub(r"^\s*(?:the\s+)?(?:evidence\s+)?(?:supported|supports?|contradicted|contradicts?)\b"
                       r"\s*(?:(?:the|this)\s+)?(?:claim)?"
                       r"(?:in\s+the\s+(?:reported|experimental)\s+setting)?\s*[:：\-]?\s*", "", answer,
                       flags=re.IGNORECASE)
    if normalized(query) in {normalized(answer), normalized(remainder)}:
        return "question_echo"
    # Cosmetic article changes still repeat the claim. Keep negations, numbers,
    # entities and word order; a set-of-words comparison would erase meaning.
    without_articles = lambda value: [word for word in re.findall(r"\w+", value.lower())
                                      if word not in {"a", "an", "the"}]
    if without_articles(query) and without_articles(query) in [without_articles(answer), without_articles(remainder)]:
        return "question_echo"
    if MISSING_FACT.search(answer):
        return "answer_reports_missing_fact"
    return None
