"""Unit tests for semantic embedder, FAISS indices, chunk aggregation, and retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.semantic.embedder import SentenceEmbedder
from src.semantic.faiss_index import FaissIndex
from src.semantic.retriever import SemanticRetriever


@pytest.fixture
def mock_corpus() -> List[dict]:
    return [
        {
            "doc_id": "doc_dl",
            "title": "Deep Learning",
            "clean_body_text": "Deep learning models use multi-layer artificial neural networks to learn representations from raw data. Backpropagation with gradient descent adjusts weights across convolutional and recurrent layers.",
        },
        {
            "doc_id": "doc_stat",
            "title": "Statistics",
            "clean_body_text": "Statistics involves hypothesis testing, confidence intervals, regression analysis, and estimating parameters of probability distributions to infer conclusions from samples.",
        },
        {
            "doc_id": "doc_cv",
            "title": "Computer Vision",
            "clean_body_text": "Computer vision algorithms process visual imagery using convolutional filters, edge detection, object segmentation, and bounding box localization.",
        },
    ]


class TestSentenceEmbedder:
    """Test embedding model wrapper, vector normalization, and caching."""

    def test_embedding_dimensions_and_norm(self) -> None:
        embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize_embeddings=True)
        texts = ["Machine learning algorithms optimize mathematical loss functions.", "Statistical inference."]
        embeddings = embedder.embed_passages(texts)

        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32

        # Verify unit L2 normalization
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    def test_query_embedding_and_bge_prefix(self) -> None:
        embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        q_vec = embedder.embed_query("gradient descent optimization")
        assert q_vec.shape == (384,)
        assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-5)

    def test_disk_caching(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "test_embeddings.npy"
        dummy_embeddings = np.random.randn(10, 384).astype(np.float32)

        SentenceEmbedder.save_cache(dummy_embeddings, cache_file)
        assert cache_file.exists()

        loaded = SentenceEmbedder.load_cache(cache_file)
        np.testing.assert_array_equal(dummy_embeddings, loaded)


class TestFaissIndexing:
    """Test exact IndexFlatIP and approximate IndexHNSWFlat search."""

    def test_exact_and_hnsw_search(self) -> None:
        dim = 16
        n_vectors = 50
        np.random.seed(42)

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        query = vectors[5].copy()  # Query identical to vector 5

        # Exact Index
        exact = FaissIndex(dimension=dim, index_type="exact")
        exact.add(vectors)
        scores_exact, idx_exact = exact.search(query, k=5)

        assert idx_exact[0] == 5
        assert np.isclose(scores_exact[0], 1.0, atol=1e-5)

        # HNSW Index
        hnsw = FaissIndex(dimension=dim, index_type="hnsw", hnsw_m=16, ef_construction=32, ef_search=32)
        hnsw.add(vectors)
        scores_hnsw, idx_hnsw = hnsw.search(query, k=5)

        assert idx_hnsw[0] == 5
        assert np.isclose(scores_hnsw[0], 1.0, atol=1e-5)


class TestSemanticRetriever:
    """Test full semantic retriever, chunk aggregation, and search."""

    def test_retriever_indexing_and_aggregation(self, mock_corpus: List[dict]) -> None:
        embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        retriever = SemanticRetriever(
            embedder=embedder,
            aggregation_strategy="max",
            chunk_size_words=25,
            chunk_overlap_words=5,
            min_chunk_words=10,
        )

        retriever.index(mock_corpus)
        assert retriever.exact_index is not None
        assert retriever.ann_index is not None

        # Search for deep learning query
        results = retriever.search("convolutional neural networks backpropagation", k=2, use_ann=False)
        assert len(results) > 0

        # doc_dl should rank first
        top_doc_id, top_score = results[0]
        assert top_doc_id == "doc_dl"
        assert top_score > 0.0

    def test_save_and_load(self, mock_corpus: List[dict], tmp_path: Path) -> None:
        index_dir = tmp_path / "semantic_index"
        embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        retriever = SemanticRetriever(embedder=embedder, chunk_size_words=50, chunk_overlap_words=10)
        retriever.index(mock_corpus)

        query = "hypothesis testing regression"
        orig_results = retriever.search(query, k=3, use_ann=False)

        retriever.save(index_dir)
        assert (index_dir / "exact.faiss").exists()
        assert (index_dir / "metadata.json").exists()

        loaded_retriever = SemanticRetriever(embedder=embedder)
        loaded_retriever.load(index_dir)

        loaded_results = loaded_retriever.search(query, k=3, use_ann=False)
        assert orig_results == loaded_results
