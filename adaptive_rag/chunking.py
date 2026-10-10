"""Deterministic local-document spans with offsets in the original source."""
from __future__ import annotations

from dataclasses import dataclass
import re

from .data import Document

BOUNDARY = re.compile(r"\n\s*\n|(?<=[.!?。！？])\s+|\n")


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source_id: str
    title: str
    text: str
    start: int
    end: int


def chunk_document(document: Document, max_chars: int = 1200,
                   overlap: int = 150) -> list[DocumentChunk]:
    """Split source content, preferring sentence boundaries within the size cap.

    No normalization or whitespace removal is applied: every chunk is exactly
    ``document.content[start:end]``. Title/newline are included in the original
    offset coordinate system, matching the existing citation audit. These are
    character limits, not a claim that every multilingual span fits192 tokens.
    """
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("Require positive max_chars and 0 <= overlap < max_chars")
    content = document.content
    if not content:
        return []
    chunks, start = [], 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content):
            # Never choose a boundary so short that overlap prevents progress.
            lower = start + max(overlap + 1, int(max_chars * .65))
            boundaries = [m.end() for m in BOUNDARY.finditer(content, lower, end)]
            if boundaries:
                end = boundaries[-1]
            else:
                space = content.rfind(" ", lower, end)
                if space >= lower:
                    end = space + 1
        chunks.append(DocumentChunk(id=f"{document.id}::span:{start}-{end}",
                                    source_id=document.id, title=document.title,
                                    text=content[start:end], start=start, end=end))
        if end == len(content):
            break
        start = end - overlap
    return chunks
