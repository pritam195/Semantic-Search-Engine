"""Production FastAPI Application for Lexical and Semantic Search Engine.

Endpoints:
- GET /search?q={query}&method=tfidf|bm25|semantic|hybrid&k={10}
- GET /compare?q={query}&k={10}
- GET /failures
- GET /health
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.service import PHASE4_FAILURE_EXAMPLES, SearchEngineService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("search_api")

service = SearchEngineService.get_instance()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all indices and model weights once at server startup."""
    logger.info("Server startup: Loading retrieval indices...")
    service.initialize()
    logger.info("Indices successfully loaded and cached in memory.")
    yield
    logger.info("Server shutdown.")


app = FastAPI(
    title="Semantic Search Engine API",
    description="Benchmarking and Serving Classical Lexical (TF-IDF, BM25) and Dense Semantic Vector Search.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---


class TermContribution(BaseModel):
    """Explainability weight for an individual query term."""

    term: str = Field(..., description="Vocabulary term appearing in both query and document.")
    weight: float = Field(..., description="Calculated contribution score (q_t * d_t).")
    percentage: float = Field(..., description="Percentage of total cosine similarity score.")


class SearchResultItem(BaseModel):
    """Single document hit with snippet and explainability metadata."""

    rank: int = Field(..., description="1-based rank position.")
    doc_id: str = Field(..., description="Canonical document identifier.")
    title: str = Field(..., description="Document page title.")
    url: str = Field(..., description="Source Wikipedia URL.")
    snippet: str = Field(..., description="Representative text snippet.")
    score: float = Field(..., description="Retrieval similarity score.")
    method: str = Field(..., description="Method used for retrieval (tfidf, bm25, semantic, hybrid).")
    top_terms: Optional[List[TermContribution]] = Field(
        None, description="Top contributing query terms for TF-IDF explainability."
    )
    best_chunk: Optional[str] = Field(
        None, description="Highest-scoring passage chunk for semantic explainability."
    )
    best_chunk_id: Optional[str] = Field(None, description="Passage chunk identifier.")


class SearchResponse(BaseModel):
    """Response model for /search endpoint."""

    query: str
    method: str
    k: int
    latency_ms: float
    total_hits: int
    results: List[SearchResultItem]


class ComparisonSummary(BaseModel):
    """High-level overlap analysis between TF-IDF and Semantic search lists."""

    overlap_count: int = Field(..., description="Number of documents shared between both methods.")
    overlap_ratio: float = Field(..., description="Fraction of top-k results that are shared.")
    overlap_doc_ids: List[str] = Field(..., description="Document IDs returned by both methods.")
    tfidf_only_doc_ids: List[str] = Field(..., description="Document IDs returned strictly by TF-IDF.")
    semantic_only_doc_ids: List[str] = Field(..., description="Document IDs returned strictly by Semantic search.")


class CompareResponse(BaseModel):
    """Response model for /compare endpoint returning all methods side by side."""

    query: str
    k: int
    total_latency_ms: float
    methods: Dict[str, SearchResponse]
    comparison: ComparisonSummary


class FailureExample(BaseModel):
    """Curated failure query from Phase 4 error analysis demonstrating algorithmic contrast."""

    id: str
    query: str
    category: str
    target_topic: str
    why_lexical_fails: str
    why_semantic_wins: str


# --- API Routes ---


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint reporting index load status and memory state."""
    return {
        "status": "healthy",
        "indexes_loaded": service.is_loaded,
        "corpus_documents": len(service.doc_ids),
        "vocab_size": len(service.tfidf.vocabulary) if service.tfidf else 0,
        "faiss_vectors": service.faiss_exact.ntotal if service.faiss_exact else 0,
        "timestamp": time.time(),
    }


@app.get("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query string"),
    method: Literal["tfidf", "bm25", "semantic", "hybrid"] = Query(
        "hybrid", description="Retrieval method to use"
    ),
    k: int = Query(10, ge=1, le=50, description="Number of results to return"),
) -> SearchResponse:
    """Retrieve top-k documents using specified method with latency and explainability."""
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if method == "tfidf":
        res = service.search_tfidf(query, k=k)
    elif method == "bm25":
        res = service.search_bm25(query, k=k)
    elif method == "semantic":
        res = service.search_semantic(query, k=k)
    elif method == "hybrid":
        res = service.search_hybrid(query, k=k)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported method: {method}")

    return SearchResponse(
        query=query,
        method=method,
        k=k,
        latency_ms=res["latency_ms"],
        total_hits=res["total_hits"],
        results=res["results"],
    )


@app.get("/compare", response_model=CompareResponse, status_code=status.HTTP_200_OK)
async def compare(
    q: str = Query(..., min_length=1, max_length=500, description="Search query string"),
    k: int = Query(10, ge=1, le=50, description="Number of top candidates per method"),
) -> CompareResponse:
    """Run all retrieval methods side-by-side with latency and overlap analytics."""
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    comparison_data = service.compare(query, k=k)

    methods_dict: Dict[str, SearchResponse] = {}
    for m_name, m_res in comparison_data["methods"].items():
        methods_dict[m_name] = SearchResponse(
            query=query,
            method=m_name,
            k=k,
            latency_ms=m_res["latency_ms"],
            total_hits=m_res["total_hits"],
            results=m_res["results"],
        )

    comp_summary = ComparisonSummary(**comparison_data["comparison"])

    return CompareResponse(
        query=query,
        k=k,
        total_latency_ms=comparison_data["total_latency_ms"],
        methods=methods_dict,
        comparison=comp_summary,
    )


@app.get("/failures", response_model=List[FailureExample], status_code=status.HTTP_200_OK)
async def get_failure_examples() -> List[FailureExample]:
    """Retrieve preloaded failure queries from Phase 4 error analysis."""
    return [FailureExample(**item) for item in PHASE4_FAILURE_EXAMPLES]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=False)
