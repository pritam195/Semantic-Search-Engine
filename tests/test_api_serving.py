"""Pytest Suite for Phase 6 FastAPI Search and Serving Endpoints (src/api/app.py).

Verifies:
1. GET /health status check and index readiness.
2. GET /search for all 4 retrieval methods (tfidf, bm25, semantic, hybrid).
3. GET /compare side-by-side execution with overlap metrics.
4. GET /failures preloaded Phase 4 error queries.
5. Explainability data: TF-IDF term contribution breakdown and semantic passage chunks.
6. Pydantic parameter validation (k limits, method literals, query validation).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.service import SearchEngineService

# Initialize test client
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def initialize_service():
    """Ensure search engine service indices are loaded before running test suite."""
    svc = SearchEngineService.get_instance()
    svc.initialize()


class TestSearchApi:
    """FastAPI endpoints integration test suite."""

    def test_health_check(self):
        """GET /health must return status healthy with loaded index counts."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["indexes_loaded"] is True
        assert data["corpus_documents"] > 0
        assert data["vocab_size"] > 0
        assert data["faiss_vectors"] > 0

    def test_search_tfidf_with_explainability(self):
        """GET /search with method=tfidf returns ranked results and term contributions."""
        resp = client.get("/search", params={"q": "machine learning algorithm", "method": "tfidf", "k": 5})
        assert resp.status_code == 200
        data = resp.json()

        assert data["query"] == "machine learning algorithm"
        assert data["method"] == "tfidf"
        assert data["k"] == 5
        assert data["latency_ms"] > 0.0
        assert len(data["results"]) == 5

        first_hit = data["results"][0]
        assert first_hit["rank"] == 1
        assert "title" in first_hit
        assert first_hit["score"] > 0.0
        assert "snippet" in first_hit
        assert len(first_hit["snippet"]) > 0

        # Verify TF-IDF term contribution explainability
        assert "top_terms" in first_hit
        top_terms = first_hit["top_terms"]
        assert len(top_terms) > 0
        assert all("term" in t and "weight" in t and "percentage" in t for t in top_terms)
        total_pct = sum(t["percentage"] for t in top_terms)
        assert 0.0 < total_pct <= 100.1

    def test_search_bm25(self):
        """GET /search with method=bm25 returns ranked results."""
        resp = client.get("/search", params={"q": "neural networks", "method": "bm25", "k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "bm25"
        assert len(data["results"]) == 5
        assert data["results"][0]["score"] > 0.0

    def test_search_semantic_with_best_chunk(self):
        """GET /search with method=semantic returns cosine scores and best matching chunk."""
        resp = client.get("/search", params={"q": "deep learning optimization", "method": "semantic", "k": 5})
        assert resp.status_code == 200
        data = resp.json()

        assert data["method"] == "semantic"
        assert len(data["results"]) == 5

        hit = data["results"][0]
        assert hit["score"] > 0.0
        assert hit["best_chunk"] is not None
        assert len(hit["best_chunk"]) > 0
        assert hit["best_chunk_id"] is not None

    def test_search_hybrid(self):
        """GET /search with method=hybrid executes score fusion across lexical and semantic."""
        resp = client.get("/search", params={"q": "support vector machines", "method": "hybrid", "k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "hybrid"
        assert len(data["results"]) == 5

    def test_compare_endpoint_side_by_side(self):
        """GET /compare returns all methods side by side with overlap analytics."""
        resp = client.get("/compare", params={"q": "margin classification algorithms", "k": 8})
        assert resp.status_code == 200
        data = resp.json()

        assert data["query"] == "margin classification algorithms"
        assert data["k"] == 8
        assert data["total_latency_ms"] > 0.0

        methods = data["methods"]
        assert "tfidf" in methods
        assert "semantic" in methods
        assert "bm25" in methods
        assert "hybrid" in methods

        comp = data["comparison"]
        assert "overlap_count" in comp
        assert "overlap_ratio" in comp
        assert "overlap_doc_ids" in comp
        assert "tfidf_only_doc_ids" in comp
        assert "semantic_only_doc_ids" in comp

        assert comp["overlap_count"] == len(comp["overlap_doc_ids"])
        assert 0.0 <= comp["overlap_ratio"] <= 1.0

    def test_failures_endpoint(self):
        """GET /failures returns curated Phase 4 failure queries."""
        resp = client.get("/failures")
        assert resp.status_code == 200
        failures = resp.json()
        assert len(failures) >= 6

        first_failure = failures[0]
        assert "query" in first_failure
        assert "category" in first_failure
        assert "why_lexical_fails" in first_failure
        assert "why_semantic_wins" in first_failure
        assert "target_topic" in first_failure

    def test_query_validation_errors(self):
        """Pydantic query validation rejects invalid methods and out-of-range k."""
        # Invalid method
        resp = client.get("/search", params={"q": "test", "method": "unknown_method"})
        assert resp.status_code == 422

        # k below minimum (ge=1)
        resp = client.get("/search", params={"q": "test", "k": 0})
        assert resp.status_code == 422

        # k above maximum (le=50)
        resp = client.get("/search", params={"q": "test", "k": 999})
        assert resp.status_code == 422
