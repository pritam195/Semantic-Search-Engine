"""Unit tests for IR metrics (Precision@k, Recall@k, MRR, MAP, nDCG@k) on known toy examples."""

from __future__ import annotations

import math
import pytest

from src.eval.metrics import (
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestMetricsAgainstToyExamples:
    """Validate from-scratch metric calculations against exact hand-computed values."""

    def test_precision_at_k(self) -> None:
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3"}

        assert precision_at_k(retrieved, relevant, k=1) == 1.0   # 1/1
        assert precision_at_k(retrieved, relevant, k=2) == 0.5   # 1/2
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2.0 / 3.0)
        assert precision_at_k(retrieved, relevant, k=5) == 0.4   # 2/5
        assert precision_at_k(retrieved, set(), k=5) == 0.0

    def test_recall_at_k(self) -> None:
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "docX", "docY"}  # 4 total relevant

        assert recall_at_k(retrieved, relevant, k=1) == 0.25  # 1/4
        assert recall_at_k(retrieved, relevant, k=3) == 0.50  # 2/4
        assert recall_at_k(retrieved, relevant, k=5) == 0.50  # 2/4
        assert recall_at_k(retrieved, set(), k=5) == 0.0

    def test_reciprocal_rank(self) -> None:
        relevant = {"target"}

        assert reciprocal_rank(["target", "d2", "d3"], relevant) == 1.0
        assert reciprocal_rank(["d1", "target", "d3"], relevant) == 0.5
        assert reciprocal_rank(["d1", "d2", "target"], relevant) == pytest.approx(1.0 / 3.0)
        assert reciprocal_rank(["d1", "d2", "d3"], relevant) == 0.0

    def test_average_precision(self) -> None:
        # doc1 (rel), doc2 (non-rel), doc3 (rel)
        # Total relevant = 2
        # P@1 = 1/1, P@3 = 2/3
        # AP = (1.0 + 2/3) / 2 = 5/6 = 0.83333...
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3"}
        expected_ap = (1.0 + (2.0 / 3.0)) / 2.0

        assert average_precision(retrieved, relevant) == pytest.approx(expected_ap, abs=1e-5)

    def test_ndcg_at_k_exact_calculation(self) -> None:
        retrieved = ["doc1", "doc2", "doc3"]
        graded = {"doc1": 2, "doc2": 0, "doc3": 1}

        # Actual DCG@3:
        # rank 1: (2^2 - 1)/log2(2) = 3/1 = 3.0
        # rank 2: 0
        # rank 3: (2^1 - 1)/log2(4) = 1/2 = 0.5
        # DCG = 3.5
        actual_dcg = dcg_at_k(retrieved, graded, k=3)
        assert actual_dcg == pytest.approx(3.5, abs=1e-5)

        # Ideal DCG@3 with grades [2, 1]:
        # rank 1: (2^2 - 1)/log2(2) = 3.0
        # rank 2: (2^1 - 1)/log2(3) = 1.0 / log2(3) = 0.63092975357
        ideal_dcg = 3.0 + (1.0 / math.log2(3.0))
        expected_ndcg = 3.5 / ideal_dcg

        assert ndcg_at_k(retrieved, graded, k=3) == pytest.approx(expected_ndcg, abs=1e-5)

    def test_ndcg_perfect_ranking_is_one(self) -> None:
        retrieved = ["doc1", "doc2", "doc3"]
        graded = {"doc1": 2, "doc2": 1, "doc3": 0}
        assert ndcg_at_k(retrieved, graded, k=3) == pytest.approx(1.0, abs=1e-5)

    def test_ndcg_empty_or_zero_relevance(self) -> None:
        retrieved = ["doc1", "doc2"]
        assert ndcg_at_k(retrieved, {}, k=2) == 0.0
        assert ndcg_at_k(retrieved, {"doc1": 0, "doc2": 0}, k=2) == 0.0
