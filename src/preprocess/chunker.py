"""Document chunking and passage splitting utility.

Splits long documents into manageable, overlapping text passages while
preserving parent document metadata for accurate passage retrieval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    """Represents a passage chunk of a parent document."""

    chunk_id: str
    doc_id: str
    title: str
    url: str
    chunk_index: int
    text: str
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextChunker:
    """Sliding-window passage chunker with word overlap."""

    def __init__(
        self,
        chunk_size_words: int = 300,
        chunk_overlap_words: int = 50,
        min_chunk_words: int = 30,
    ) -> None:
        if chunk_overlap_words >= chunk_size_words:
            raise ValueError("chunk_overlap_words must be strictly less than chunk_size_words")
        self.chunk_size = chunk_size_words
        self.chunk_overlap = chunk_overlap_words
        self.min_chunk_words = min_chunk_words
        self.step_size = chunk_size_words - chunk_overlap_words

    def chunk_document(
        self,
        doc_id: str,
        title: str,
        url: str,
        text: str,
    ) -> List[DocumentChunk]:
        """Split a document's clean body text into overlapping chunks.

        Args:
            doc_id: Parent document unique identifier.
            title: Parent document title.
            url: Parent document URL.
            text: Full clean body text.

        Returns:
            List of DocumentChunk instances.
        """
        words = text.split()
        if not words:
            return []

        # If document is short enough, emit as a single chunk
        if len(words) <= self.chunk_size:
            if len(words) >= self.min_chunk_words:
                return [
                    DocumentChunk(
                        chunk_id=f"{doc_id}#0",
                        doc_id=doc_id,
                        title=title,
                        url=url,
                        chunk_index=0,
                        text=" ".join(words),
                        word_count=len(words),
                    )
                ]
            return []

        chunks: List[DocumentChunk] = []
        chunk_idx = 0
        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            if len(chunk_words) < self.min_chunk_words and chunks:
                # Merge tiny trailing remainder into the previous chunk
                prev = chunks[-1]
                merged_text = prev.text + " " + " ".join(chunk_words)
                chunks[-1] = DocumentChunk(
                    chunk_id=prev.chunk_id,
                    doc_id=prev.doc_id,
                    title=prev.title,
                    url=prev.url,
                    chunk_index=prev.chunk_index,
                    text=merged_text,
                    word_count=len(merged_text.split()),
                )
                break

            chunk_text = " ".join(chunk_words)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}#{chunk_idx}",
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    word_count=len(chunk_words),
                )
            )
            chunk_idx += 1
            i += self.step_size

        return chunks
