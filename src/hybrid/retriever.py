"""Unified Hybrid Retriever combining Lexical (BM25/TF-IDF) and Semantic (Dense) search.

Supports:
- Reciprocal Rank Fusion (RRF)
- Min-Max Normalized Weighted Score Fusion
- Optional Cross-Encoder Neural Reranking (top 50 candidates)
- BaseRetriever API compatibility
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.hybrid.fusion import reciprocal_rank_fusion, score_fusion
from src.hybrid.reranker import CrossEncoderReranker
from src.lexical.base import BaseRetriever, SearchResult

logger = logging.getLogger("hybrid_retriever")


class HybridRetriever(BaseRetriever):
    """Combines lexical and dense semantic retrievers with rank/score fusion and reranking."""

    def __init__(
        self,
        lexical_retriever: BaseRetriever,
        semantic_retriever: BaseRetriever,
        fusion_mode: str = "rrf",
        alpha: float = 0.5,
        k_rrf: int = 60,
        candidate_k: int = 50,
        reranker: Optional[CrossEncoderReranker] = None,
        doc_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            lexical_retriever: BM25 or TF-IDF retriever instance.
            semantic_retriever: Dense semantic retriever instance (MiniLM or BGE).
            fusion_mode: 'rrf' (Reciprocal Rank Fusion) or 'score_fusion' (weighted linear fusion).
            alpha: Weight for lexical retriever in score fusion (semantic weight is 1 - alpha).
            k_rrf: Smoothing parameter for RRF (default: 60).
            candidate_k: Number of candidates to retrieve from each source before fusion.
            reranker: Optional CrossEncoderReranker instance to rerank top candidates.
            doc_lookup: Optional corpus dictionary doc_id -> doc dict for reranker text context.
        """
        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever
        self.fusion_mode = fusion_mode.lower()
        self.alpha = float(alpha)
        self.k_rrf = int(k_rrf)
        self.candidate_k = int(candidate_k)
        self.reranker = reranker
        self.doc_lookup = doc_lookup or {}

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Retrieve and rank documents using hybrid fusion and optional reranking.

        Args:
            query: Search query string.
            k: Number of top results to return.

        Returns:
            List of (doc_id, score) pairs sorted descending by score.
        """
        # 1. First-stage retrieval from both lexical and semantic retrievers
        lexical_results = self.lexical_retriever.search(query, k=self.candidate_k)
        semantic_results = self.semantic_retriever.search(query, k=self.candidate_k)

        # 2. Combine using chosen fusion strategy
        if self.fusion_mode == "score_fusion":
            # Weighted score fusion with min-max normalization
            weights = [self.alpha, 1.0 - self.alpha]
            fused = score_fusion(
                scored_lists=[lexical_results, semantic_results],
                weights=weights,
                normalize=True,
                top_k=max(k, self.candidate_k),
            )
        else:
            # Reciprocal Rank Fusion (RRF)
            fused = reciprocal_rank_fusion(
                ranked_lists=[lexical_results, semantic_results],
                k_rrf=self.k_rrf,
                top_k=max(k, self.candidate_k),
            )

        # 3. Optional Second-stage Cross-Encoder Reranking
        if self.reranker is not None and self.doc_lookup:
            rerank_candidates = [doc_id for doc_id, _ in fused[:self.candidate_k]]
            final_results = self.reranker.rerank_doc_lookup(
                query=query,
                candidate_ids=rerank_candidates,
                doc_lookup=self.doc_lookup,
                top_k=k,
            )
            return final_results

        return fused[:k]

    def search_with_metadata(
        self,
        query: str,
        k: int = 10,
    ) -> List[SearchResult]:
        """Search and format results as SearchResult objects with metadata."""
        ranked_pairs = self.search(query, k=k)
        results: List[SearchResult] = []

        for rank, (doc_id, score) in enumerate(ranked_pairs, start=1):
            doc = self.doc_lookup.get(doc_id, {})
            title = doc.get("title", "")
            url = doc.get("url", "")
            raw_text = doc.get("clean_body_text") or doc.get("raw_text", "")
            snippet = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text

            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=float(score),
                    rank=rank,
                    title=title,
                    url=url,
                    snippet=snippet,
                    metadata={"doc_id": doc_id, "score": score, "fusion_mode": self.fusion_mode},
                )
            )

        return results

    def index(self, corpus: List[Dict[str, Any]]) -> None:
        """Index corpus in both lexical and semantic retrievers."""
        self.doc_lookup = {d.get("doc_id", ""): d for d in corpus}
        self.lexical_retriever.index(corpus)
        self.semantic_retriever.index(corpus)

    def save(self, directory: Any) -> None:
        """Save underlying retrievers."""
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        self.lexical_retriever.save(p / "lexical")
        self.semantic_retriever.save(p / "semantic")

    def load(self, directory: Any) -> None:
        """Load underlying retrievers."""
        p = Path(directory)
        self.lexical_retriever.load(p / "lexical")
        self.semantic_retriever.load(p / "semantic")
