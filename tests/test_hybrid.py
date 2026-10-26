"""Unit tests for Hybrid Retrieval, Fusion algorithms, and Reranking."""

from __future__ import annotations

import math
import pytest

from src.hybrid.fusion import min_max_normalize, reciprocal_rank_fusion, score_fusion
from src.hybrid.reranker import CrossEncoderReranker
from src.hybrid.retriever import HybridRetriever
from src.lexical.base import BaseRetriever


class MockRetriever(BaseRetriever):
    """Simple mock retriever for deterministic unit testing."""

    def __init__(self, canned_results: dict[str, list[tuple[str, float]]]):
        self.canned = canned_results

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self.canned.get(query, [])[:k]

    def index(self, corpus: list[dict]) -> None:
        pass

    def search_with_metadata(self, query: str, k: int = 10) -> list:
        return []

    def save(self, directory) -> None:
        pass

    def load(self, directory) -> None:
        pass


class TestFusionAlgorithms:
    """Test mathematical correctness of fusion algorithms."""

    def test_min_max_normalize_basic(self) -> None:
        items = [("d1", 10.0), ("d2", 20.0), ("d3", 30.0)]
        normalized = min_max_normalize(items)
        assert normalized["d1"] == pytest.approx(0.0, abs=1e-6)
        assert normalized["d2"] == pytest.approx(0.5, abs=1e-6)
        assert normalized["d3"] == pytest.approx(1.0, abs=1e-6)

    def test_min_max_normalize_identical_scores(self) -> None:
        items = [("d1", 5.0), ("d2", 5.0), ("d3", 5.0)]
        normalized = min_max_normalize(items)
        assert normalized["d1"] == 1.0
        assert normalized["d2"] == 1.0
        assert normalized["d3"] == 1.0

    def test_min_max_normalize_empty(self) -> None:
        assert min_max_normalize([]) == {}

    def test_reciprocal_rank_fusion_exact_math(self) -> None:
        # List 1: [docA, docB, docC]
        # List 2: [docA, docD, docB]
        list1 = [("docA", 10.0), ("docB", 5.0), ("docC", 2.0)]
        list2 = [("docA", 0.9), ("docD", 0.7), ("docB", 0.4)]

        # With k_rrf = 60:
        # docA: 1/(60+1) + 1/(60+1) = 2/61 = 0.0327868852
        # docB: 1/(60+2) + 1/(60+3) = 1/62 + 1/63 = 0.0320020481
        # docC: 1/(60+3) = 1/63 = 0.0158730158
        # docD: 1/(60+2) = 1/62 = 0.0161290322
        fused = reciprocal_rank_fusion([list1, list2], k_rrf=60)

        assert len(fused) == 4
        doc_ids = [d for d, _ in fused]
        assert doc_ids == ["docA", "docB", "docD", "docC"]

        scores = dict(fused)
        assert scores["docA"] == pytest.approx(2.0 / 61.0, abs=1e-6)
        assert scores["docB"] == pytest.approx((1.0 / 62.0) + (1.0 / 63.0), abs=1e-6)
        assert scores["docD"] == pytest.approx(1.0 / 62.0, abs=1e-6)
        assert scores["docC"] == pytest.approx(1.0 / 63.0, abs=1e-6)

    def test_score_fusion_with_weights(self) -> None:
        # List 1: d1=10, d2=0 -> norm: d1=1.0, d2=0.0
        # List 2: d2=10, d1=0 -> norm: d2=1.0, d1=0.0
        list1 = [("d1", 10.0), ("d2", 0.0)]
        list2 = [("d1", 0.0), ("d2", 10.0)]

        # Weight: alpha=0.7 for list1, 0.3 for list2
        fused = score_fusion([list1, list2], weights=[0.7, 0.3], normalize=True)
        scores = dict(fused)

        # d1: 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        # d2: 0.7 * 0.0 + 0.3 * 1.0 = 0.3
        assert scores["d1"] == pytest.approx(0.7, abs=1e-5)
        assert scores["d2"] == pytest.approx(0.3, abs=1e-5)
        assert fused[0][0] == "d1"
        assert fused[1][0] == "d2"


class TestHybridRetrieverIntegration:
    """Test HybridRetriever class integration."""

    def test_hybrid_retriever_search_rrf(self) -> None:
        ret_lex = MockRetriever({"q1": [("doc1", 8.0), ("doc2", 4.0)]})
        ret_sem = MockRetriever({"q1": [("doc2", 0.95), ("doc3", 0.85)]})

        hybrid = HybridRetriever(lexical_retriever=ret_lex, semantic_retriever=ret_sem, fusion_mode="rrf", k_rrf=60)
        results = hybrid.search("q1", k=3)

        assert len(results) == 3
        # doc2 appears in both lists (rank 2 in lex, rank 1 in sem) -> 1/62 + 1/61 = 0.0325
        # doc1 appears rank 1 in lex -> 1/61 = 0.01639
        # doc3 appears rank 2 in sem -> 1/62 = 0.01613
        assert results[0][0] == "doc2"
        assert results[1][0] == "doc1"
        assert results[2][0] == "doc3"

    def test_hybrid_retriever_search_score_fusion(self) -> None:
        ret_lex = MockRetriever({"q1": [("doc1", 10.0), ("doc2", 0.0)]})
        ret_sem = MockRetriever({"q1": [("doc1", 0.0), ("doc2", 1.0)]})

        hybrid = HybridRetriever(
            lexical_retriever=ret_lex,
            semantic_retriever=ret_sem,
            fusion_mode="score_fusion",
            alpha=0.8,
        )
        results = hybrid.search("q1", k=2)

        # doc1: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        # doc2: 0.8 * 0.0 + 0.2 * 1.0 = 0.2
        assert results[0][0] == "doc1"
        assert results[1][0] == "doc2"
        assert results[0][1] == pytest.approx(0.8, abs=1e-5)
        assert results[1][1] == pytest.approx(0.2, abs=1e-5)


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker functionality."""

    def test_cross_encoder_reranking_accuracy(self) -> None:
        reranker = CrossEncoderReranker()
        query = "gradient descent learning rate"
        candidates = [
            ("d_irrelevant", "The recipe calls for two cups of flour and butter."),
            ("d_relevant", "In optimization algorithms, gradient descent adjusts parameters using the learning rate."),
        ]

        reranked = reranker.rerank(query, candidates, top_k=2)
        assert len(reranked) == 2
        assert reranked[0][0] == "d_relevant"
        assert reranked[1][0] == "d_irrelevant"
        assert reranked[0][1] > reranked[1][1]


class TestQuerySplitting:
    """Test train/dev and test split generation."""

    def test_stratified_split_properties(self) -> None:
        from src.eval.split import create_stratified_split

        mock_queries = [
            {"query_id": f"k{i}", "query_type": "keyword"} for i in range(10)
        ] + [
            {"query_id": f"q{i}", "query_type": "natural_language_question"} for i in range(10)
        ] + [
            {"query_id": f"p{i}", "query_type": "paraphrase_vocab_mismatch"} for i in range(10)
        ]

        dev_q, test_q = create_stratified_split(mock_queries, dev_ratio=0.5, seed=42)

        # Check disjoint
        dev_ids = {q["query_id"] for q in dev_q}
        test_ids = {q["query_id"] for q in test_q}
        assert dev_ids.isdisjoint(test_ids)
        assert len(dev_ids) + len(test_ids) == 30

        # Check stratification
        dev_types = [q["query_type"] for q in dev_q]
        test_types = [q["query_type"] for q in test_q]
        assert dev_types.count("keyword") == 5
        assert dev_types.count("natural_language_question") == 5
        assert dev_types.count("paraphrase_vocab_mismatch") == 5
        assert test_types.count("keyword") == 5
        assert test_types.count("natural_language_question") == 5
        assert test_types.count("paraphrase_vocab_mismatch") == 5
