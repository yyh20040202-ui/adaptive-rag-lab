import pytest
from adaptive_rag.data import Document, grouped_split


def test_connected_document_groups_do_not_cross_validation():
    qrels = {"a": {"paper1": 1}, "b": {"paper1": 1, "paper2": 1}, "c": {"paper2": 1},
             "d": {"paper3": 1}, "e": {"paper4": 1}, "f": {"paper5": 1}}
    train, val = grouped_split(qrels, seed=7, validation_fraction=.3)
    assert sorted(train+val) == sorted(qrels)
    assert {"a", "b", "c"} <= set(train) or {"a", "b", "c"} <= set(val)
    assert not {p for q in train for p in qrels[q]} & {p for q in val for p in qrels[q]}
    assert grouped_split(qrels, seed=7, validation_fraction=.3) == (train, val)


def test_unsplittable_labels_fail_explicitly():
    with pytest.raises(ValueError):
        grouped_split({"a": {"same": 1}, "b": {"same": 1}})


def test_document_content_preserves_quote_offsets():
    doc = Document("d", "Policy", "Beijing limit is CNY 500.")
    assert doc.content[7:14] == "Beijing"
