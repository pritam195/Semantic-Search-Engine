"""Unit tests for text preprocessing, tokenization, and passage chunking."""

from __future__ import annotations

import pytest

from src.preprocess.chunker import TextChunker
from src.preprocess.tokenizer import TextPreprocessor


class TestTextPreprocessor:
    """Test text cleaning, tokenization, stopword removal, and stemming."""

    def test_lowercasing_and_punctuation(self) -> None:
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            stopwords=None,
            stemmer="none",
        )
        text = "Deep-Learning: Neural Networks & Artificial Intelligence!"
        tokens = preprocessor.preprocess(text)
        assert "deeplearning" in tokens or "deep" in tokens
        assert "neural" in tokens
        assert "networks" in tokens
        assert "artificial" in tokens
        assert "intelligence" in tokens

    def test_stopword_removal(self) -> None:
        preprocessor = TextPreprocessor(
            lowercase=True,
            stopwords="english",
            stemmer="none",
        )
        text = "This is a simple test of the search engine network"
        tokens = preprocessor.preprocess(text)
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "of" not in tokens
        assert "the" not in tokens
        assert "simple" in tokens
        assert "test" in tokens
        assert "search" in tokens
        assert "engine" in tokens
        assert "network" in tokens

    def test_porter_stemmer(self) -> None:
        preprocessor = TextPreprocessor(
            lowercase=True,
            stopwords="english",
            stemmer="porter",
        )
        text = "Machine learning algorithms optimize mathematical functions"
        tokens = preprocessor.preprocess(text)
        # Porter stemmer rules: learning -> learn, algorithms -> algorithm, optimize -> optim
        assert "machin" in tokens
        assert "learn" in tokens
        assert "algorithm" in tokens
        assert "optim" in tokens

    def test_empty_and_whitespace_input(self) -> None:
        preprocessor = TextPreprocessor()
        assert preprocessor.preprocess("") == []
        assert preprocessor.preprocess("   \n\t  ") == []


class TestTextChunker:
    """Test passage chunking and metadata preservation."""

    def test_short_document_single_chunk(self) -> None:
        chunker = TextChunker(chunk_size_words=100, chunk_overlap_words=20, min_chunk_words=10)
        text = " ".join(["word"] * 50)
        chunks = chunker.chunk_document("doc1", "Test Title", "http://example.com/1", text)

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc1#0"
        assert chunks[0].doc_id == "doc1"
        assert chunks[0].word_count == 50

    def test_long_document_multi_chunk_with_overlap(self) -> None:
        chunker = TextChunker(chunk_size_words=100, chunk_overlap_words=20, min_chunk_words=10)
        words = [f"w{i}" for i in range(250)]
        text = " ".join(words)

        chunks = chunker.chunk_document("doc2", "Long Document", "http://example.com/2", text)
        assert len(chunks) > 1

        # Check step size: chunk_size - overlap = 80
        # Chunk 0: words 0..100
        # Chunk 1: words 80..180 (starts with w80)
        c0_words = chunks[0].text.split()
        c1_words = chunks[1].text.split()
        assert c0_words[0] == "w0"
        assert c1_words[0] == "w80"

        # Check overlap
        assert c0_words[-20:] == c1_words[:20]

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            TextChunker(chunk_size_words=100, chunk_overlap_words=100)
