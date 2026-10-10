import pytest

from adaptive_rag.chunking import chunk_document
from adaptive_rag.data import Document


def test_long_source_has_no_gaps_and_exact_original_offsets():
    source = Document("source", "会议记录", "第一页内容。\n" * 180 + "关键报销额度在文末：500元。")
    chunks = chunk_document(source, max_chars=180, overlap=25)
    assert len(chunks) > 2
    assert chunks == chunk_document(source, max_chars=180, overlap=25)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert chunks[0].start == 0 and chunks[-1].end == len(source.content)
    assert all(0 < len(chunk.text) <= 180 for chunk in chunks)
    assert all(chunk.source_id == source.id and chunk.title == source.title for chunk in chunks)
    assert all(chunk.text == source.content[chunk.start:chunk.end] for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start == previous.end - 25
        assert current.end > previous.end
    assert "500元" in chunks[-1].text


def test_sentence_boundary_is_preferred_without_normalizing_whitespace():
    source = Document("x", "", "A" * 70 + ".\n  " + "B" * 150)
    chunks = chunk_document(source, max_chars=100, overlap=10)
    assert chunks[0].end == 74
    assert chunks[0].text.endswith(".\n  ")
    assert chunks[1].start == 64


def test_empty_short_and_zero_overlap_sources():
    assert chunk_document(Document("empty", "", "")) == []
    source = Document("short", "Title", "正文")
    chunks = chunk_document(source)
    assert len(chunks) == 1 and chunks[0].text == source.content
    source = Document("no-space", "", "字" * 501)
    chunks = chunk_document(source, max_chars=100, overlap=0)
    assert "".join(chunk.text for chunk in chunks) == source.content
    assert len(chunks) == 6


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 101)])
def test_invalid_settings_cannot_loop_without_progress(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(Document("x", "", "long text"), max_chars=size, overlap=overlap)
