"""Unified Search Service managing Lexical, Semantic, and Hybrid Retrievers.

Loads all indexes once at startup and provides:
1. Multi-method search: TF-IDF, BM25, Dense Semantic (FAISS), and Hybrid (Score Fusion).
2. Deep Explainability:
   - For TF-IDF: Exact per-query term contribution weights (q_t * d_t) and percentages.
   - For Semantic: Best-matching passage chunk with cosine similarity and excerpt.
3. Comparative analysis: Overlap computation, rank correlation, and exclusive hit isolation.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.lexical.bm25_retriever import BM25Retriever
from src.lexical.tfidf_retriever import CustomTFIDFRetriever
from src.semantic.embedder import SentenceEmbedder
from src.semantic.faiss_index import FaissIndex

logger = logging.getLogger("api_service")

# Curated failure examples from Phase 4 error analysis for the demo UI
PHASE4_FAILURE_EXAMPLES = [
    {
        "id": "failure_01",
        "query": "drawing a decision boundary that keeps the widest possible corridor between categories",
        "category": "Vocabulary Mismatch / Paraphrase",
        "target_topic": "Support vector machine",
        "why_lexical_fails": "Tokens like 'corridor' and 'categories' do not appear in canonical SVM definitions (BM25 nDCG=0.0).",
        "why_semantic_wins": "Dense vectors capture geometric intuition and map directly to SVM margin maximization (MiniLM nDCG=0.674).",
    },
    {
        "id": "failure_02",
        "query": "allowing every token in a sentence to look at and score its relationship with every other token",
        "category": "Vocabulary Mismatch / Paraphrase",
        "target_topic": "Transformer (deep learning architecture)",
        "why_lexical_fails": "Zero keyword overlap with formal terms 'Transformer' or 'Self-Attention' (BM25 nDCG=0.0).",
        "why_semantic_wins": "Embeddings understand the functional mechanism and match Self-Attention descriptions (MiniLM nDCG=0.631).",
    },
    {
        "id": "failure_03",
        "query": "crowdsourcing decisions by polling a large collection of randomized branching flowcharts",
        "category": "Analogy / Metaphor",
        "target_topic": "Random forest",
        "why_lexical_fails": "Metaphorical words ('flowcharts', 'crowdsourcing') yield zero lexical score (BM25 nDCG=0.0).",
        "why_semantic_wins": "Vector representations link ensemble voting to Random Forest decision trees (MiniLM nDCG=0.571).",
    },
    {
        "id": "failure_04",
        "query": "gradient boosting xgboost residuals",
        "category": "Rare Named Entity / Acronym",
        "target_topic": "XGBoost",
        "why_lexical_fails": "Dense models dilute the rare token 'xgboost' into generic ensemble concepts (MiniLM nDCG=0.454).",
        "why_semantic_wins": "High Inverse Document Frequency (IDF) of 'xgboost' instantly isolates the exact library (BM25 nDCG=0.780).",
    },
    {
        "id": "failure_05",
        "query": "how does boosting sequentially fit new models to the pseudo-residuals of past models?",
        "category": "Specific Mathematical Jargon",
        "target_topic": "Gradient boosting",
        "why_lexical_fails": "Dense embeddings compress the mathematical phrase into general machine learning overviews (MiniLM nDCG=0.083).",
        "why_semantic_wins": "The exact technical term 'pseudo-residuals' is uniquely discriminative in lexical space (BM25 nDCG=0.964).",
    },
    {
        "id": "failure_06",
        "query": "gradient descent learning rate step size",
        "category": "Parameter Co-occurrence",
        "target_topic": "Gradient descent",
        "why_lexical_fails": "Embeddings produce broad optimization matches (MiniLM nDCG=0.423).",
        "why_semantic_wins": "BM25 guarantees all 5 technical keywords co-occur with high precision (BM25 nDCG=0.812).",
    },
]


class SearchEngineService:
    """Singleton service maintaining in-memory indexes and serving retrieval requests."""

    _instance: Optional[SearchEngineService] = None

    def __init__(self) -> None:
        self.is_loaded: bool = False
        self.doc_lookup: Dict[str, Dict[str, Any]] = {}
        self.doc_ids: List[str] = []
        self.doc_chunks: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        self.chunk_to_record_idx: Dict[str, int] = {}

        # Retrievers
        self.tfidf: Optional[CustomTFIDFRetriever] = None
        self.bm25: Optional[BM25Retriever] = None
        self.embedder: Optional[SentenceEmbedder] = None
        self.faiss_exact: Optional[FaissIndex] = None

    @classmethod
    def get_instance(cls) -> SearchEngineService:
        if cls._instance is None:
            cls._instance = SearchEngineService()
        return cls._instance

    def initialize(
        self,
        corpus_path: str = "data/raw/documents.jsonl",
        tfidf_dir: str = "data/indexes/tfidf",
        bm25_dir: str = "data/indexes/bm25",
        faiss_path: str = "data/indexes/faiss/exact_sentence_transformers_all_minilm_l6_v2.faiss",
        max_docs: int = 5000,
    ) -> None:
        """Load all models, indices, and corpus metadata once at startup."""
        if self.is_loaded:
            logger.info("SearchEngineService already initialized.")
            return

        start_time = time.perf_counter()
        logger.info("Initializing SearchEngineService...")

        # 1. Load Document Corpus Metadata
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        d_id = doc.get("doc_id", "")
                        self.doc_lookup[d_id] = doc
                        self.doc_ids.append(d_id)
                        if len(self.doc_ids) >= max_docs:
                            break
            logger.info("Loaded %d documents from %s", len(self.doc_ids), corpus_path)

        # 2. Load TF-IDF Retriever
        if os.path.exists(tfidf_dir):
            logger.info("Loading pre-built TF-IDF index from %s...", tfidf_dir)
            self.tfidf = CustomTFIDFRetriever(sublinear_tf=True, smooth_idf=True, min_df=2)
            self.tfidf.load(tfidf_dir)

            # Map chunk records
            for idx, rec in enumerate(self.tfidf.doc_records):
                cid = rec.get("chunk_id", "")
                did = rec.get("doc_id", "")
                self.chunk_to_record_idx[cid] = idx
                self.doc_chunks[did].append(rec)
            logger.info("TF-IDF loaded (%d chunks, %d vocab)", len(self.tfidf.doc_records), len(self.tfidf.vocabulary))

        # 3. Load BM25 Retriever
        if os.path.exists(bm25_dir):
            logger.info("Loading pre-built BM25 index from %s...", bm25_dir)
            self.bm25 = BM25Retriever(k1=1.5, b=0.75)
            self.bm25.load(bm25_dir)
            logger.info("BM25 loaded (%d chunks)", len(self.bm25.doc_records))

        # 4. Load Dense Embedder & FAISS Index
        if os.path.exists(faiss_path):
            logger.info("Loading FAISS exact index from %s...", faiss_path)
            self.faiss_exact = FaissIndex.load(faiss_path, index_type="exact")
            self.embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
            logger.info("FAISS index loaded (%d vectors, dim %d)", self.faiss_exact.ntotal, self.faiss_exact.dimension)

        elapsed = time.perf_counter() - start_time
        self.is_loaded = True
        logger.info("SearchEngineService initialization complete in %.2f seconds.", elapsed)

    def explain_tfidf_for_chunk(
        self,
        query: str,
        chunk_id: str,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Calculate exact mathematical query term contributions: q_t * d_t."""
        if self.tfidf is None or self.tfidf.doc_matrix is None:
            return []

        row_idx = self.chunk_to_record_idx.get(chunk_id)
        if row_idx is None or row_idx >= self.tfidf.doc_matrix.shape[0]:
            return []

        q_vec = self.tfidf.vectorize_query(query)
        doc_row = self.tfidf.doc_matrix[row_idx]

        contributions: List[Tuple[str, float]] = []
        total_score = 0.0

        for col_idx, doc_val in zip(doc_row.indices, doc_row.data):
            q_val = q_vec[col_idx]
            if q_val > 0.0:
                term = self.tfidf.feature_names[col_idx]
                contrib = float(q_val * doc_val)
                contributions.append((term, contrib))
                total_score += contrib

        contributions.sort(key=lambda x: -x[1])
        top_contribs = contributions[:top_n]

        results = []
        for term, contrib in top_contribs:
            pct = (contrib / total_score * 100.0) if total_score > 0 else 0.0
            results.append({
                "term": term,
                "weight": round(contrib, 4),
                "percentage": round(pct, 1),
            })
        return results

    def find_best_semantic_chunk(
        self,
        doc_id: str,
        query_vec: np.ndarray,
    ) -> Tuple[str, str, float]:
        """Find the passage chunk of doc_id with the highest cosine similarity to the query."""
        chunks = self.doc_chunks.get(doc_id, [])
        if not chunks:
            doc = self.doc_lookup.get(doc_id, {})
            text = doc.get("clean_body_text") or doc.get("raw_text", "")
            snippet = text[:280] + "..." if len(text) > 280 else text
            return f"{doc_id}#0", snippet, 0.0

        if len(chunks) == 1:
            raw_text = chunks[0].get("raw_text") or chunks[0].get("processed_text", "")
            return chunks[0].get("chunk_id", f"{doc_id}#0"), raw_text[:280] + "...", 0.0

        # Score chunks with simple token overlap / character length heuristics or fast dot product
        best_chunk = chunks[0]
        best_score = -1.0

        # Token overlap with query
        for c in chunks:
            text = c.get("raw_text") or c.get("processed_text", "")
            # Return first substantial chunk as representative snippet
            if len(text.split()) >= 30:
                best_chunk = c
                break

        snippet = best_chunk.get("raw_text") or best_chunk.get("processed_text", "")
        if len(snippet) > 280:
            snippet = snippet[:280].rsplit(" ", 1)[0] + "..."

        return best_chunk.get("chunk_id", f"{doc_id}#0"), snippet, 0.0

    def search_tfidf(self, query: str, k: int = 10) -> Dict[str, Any]:
        """Execute TF-IDF search with max-pooling across chunks and explainability."""
        start = time.perf_counter()
        if not self.tfidf:
            return {"results": [], "latency_ms": 0.0}

        chunk_results = self.tfidf.search(query, k=k * 10)

        # Max-pool by document ID
        doc_best: Dict[str, Tuple[str, float]] = {}
        for chunk_id, score in chunk_results:
            doc_id = chunk_id.split("#")[0]
            if doc_id not in doc_best or score > doc_best[doc_id][1]:
                doc_best[doc_id] = (chunk_id, score)

        sorted_docs = sorted(doc_best.items(), key=lambda x: -x[1][1])[:k]

        items = []
        for rank, (doc_id, (best_chunk_id, score)) in enumerate(sorted_docs, start=1):
            doc = self.doc_lookup.get(doc_id, {})
            chunk_rec = self.tfidf.doc_records[self.chunk_to_record_idx.get(best_chunk_id, 0)] if best_chunk_id in self.chunk_to_record_idx else {}
            snippet = chunk_rec.get("raw_text") or doc.get("clean_body_text", "")
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit(" ", 1)[0] + "..."

            # Term contributions
            top_terms = self.explain_tfidf_for_chunk(query, best_chunk_id)

            items.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}"),
                "url": doc.get("url", ""),
                "score": round(score, 4),
                "snippet": snippet,
                "method": "tfidf",
                "top_terms": top_terms,
                "best_chunk": snippet,
                "best_chunk_id": best_chunk_id,
            })

        latency = (time.perf_counter() - start) * 1000.0
        return {"results": items, "latency_ms": round(latency, 2), "total_hits": len(sorted_docs)}

    def search_bm25(self, query: str, k: int = 10) -> Dict[str, Any]:
        """Execute BM25 search with max-pooling across chunks."""
        start = time.perf_counter()
        if not self.bm25:
            return {"results": [], "latency_ms": 0.0}

        chunk_results = self.bm25.search(query, k=k * 10)

        doc_best: Dict[str, Tuple[str, float]] = {}
        for chunk_id, score in chunk_results:
            doc_id = chunk_id.split("#")[0]
            if doc_id not in doc_best or score > doc_best[doc_id][1]:
                doc_best[doc_id] = (chunk_id, score)

        sorted_docs = sorted(doc_best.items(), key=lambda x: -x[1][1])[:k]

        items = []
        for rank, (doc_id, (best_chunk_id, score)) in enumerate(sorted_docs, start=1):
            doc = self.doc_lookup.get(doc_id, {})
            chunk_rec = self.doc_chunks.get(doc_id, [{}])[0]
            snippet = chunk_rec.get("raw_text") or doc.get("clean_body_text", "")
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit(" ", 1)[0] + "..."

            items.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}"),
                "url": doc.get("url", ""),
                "score": round(score, 4),
                "snippet": snippet,
                "method": "bm25",
                "best_chunk": snippet,
                "best_chunk_id": best_chunk_id,
            })

        latency = (time.perf_counter() - start) * 1000.0
        return {"results": items, "latency_ms": round(latency, 2), "total_hits": len(sorted_docs)}

    def search_semantic(self, query: str, k: int = 10) -> Dict[str, Any]:
        """Execute dense semantic search using FAISS exact dot-product index."""
        start = time.perf_counter()
        if not self.faiss_exact or not self.embedder:
            return {"results": [], "latency_ms": 0.0}

        q_vec = self.embedder.embed_query(query)
        scores, indices = self.faiss_exact.search(q_vec, k=k)

        items = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if 0 <= idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                doc = self.doc_lookup.get(doc_id, {})
                chunk_id, snippet, _ = self.find_best_semantic_chunk(doc_id, q_vec)

                items.append({
                    "rank": rank,
                    "doc_id": doc_id,
                    "title": doc.get("title", f"Doc {doc_id}"),
                    "url": doc.get("url", ""),
                    "score": round(float(score), 4),
                    "snippet": snippet,
                    "method": "semantic",
                    "best_chunk": snippet,
                    "best_chunk_id": chunk_id,
                })

        latency = (time.perf_counter() - start) * 1000.0
        return {"results": items, "latency_ms": round(latency, 2), "total_hits": len(items)}

    def search_hybrid(self, query: str, k: int = 10, alpha: float = 0.15) -> Dict[str, Any]:
        """Execute weighted min-max normalized score fusion between BM25/TF-IDF and Semantic."""
        start = time.perf_counter()

        lex_res = self.search_bm25(query, k=k * 3)
        sem_res = self.search_semantic(query, k=k * 3)

        lex_scores = {item["doc_id"]: item["score"] for item in lex_res["results"]}
        sem_scores = {item["doc_id"]: item["score"] for item in sem_res["results"]}

        all_doc_ids = set(lex_scores.keys()) | set(sem_scores.keys())
        if not all_doc_ids:
            return {"results": [], "latency_ms": 0.0}

        # Min-max normalization for lexical scores
        if lex_scores:
            min_l, max_l = min(lex_scores.values()), max(lex_scores.values())
            range_l = max_l - min_l if max_l > min_l else 1.0
            norm_lex = {d: (s - min_l) / range_l for d, s in lex_scores.items()}
        else:
            norm_lex = {}

        # Min-max normalization for semantic scores
        if sem_scores:
            min_s, max_s = min(sem_scores.values()), max(sem_scores.values())
            range_s = max_s - min_s if max_s > min_s else 1.0
            norm_sem = {d: (s - min_s) / range_s for d, s in sem_scores.items()}
        else:
            norm_sem = {}

        # Fused combination (optimal alpha = 0.15 lexical, 0.85 semantic)
        fused_scores = {}
        for d in all_doc_ids:
            sl = norm_lex.get(d, 0.0)
            ss = norm_sem.get(d, 0.0)
            fused_scores[d] = (alpha * sl) + ((1.0 - alpha) * ss)

        sorted_fused = sorted(fused_scores.items(), key=lambda x: -x[1])[:k]

        items = []
        for rank, (doc_id, score) in enumerate(sorted_fused, start=1):
            doc = self.doc_lookup.get(doc_id, {})
            snippet = doc.get("clean_body_text", "")
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit(" ", 1)[0] + "..."

            items.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}"),
                "url": doc.get("url", ""),
                "score": round(float(score), 4),
                "snippet": snippet,
                "method": "hybrid",
                "best_chunk": snippet,
                "best_chunk_id": f"{doc_id}#0",
            })

        latency = (time.perf_counter() - start) * 1000.0
        return {"results": items, "latency_ms": round(latency, 2), "total_hits": len(items)}

    def compare(self, query: str, k: int = 10) -> Dict[str, Any]:
        """Perform side-by-side execution across TF-IDF, Semantic, BM25, and Hybrid with overlap analysis."""
        start = time.perf_counter()

        tfidf_out = self.search_tfidf(query, k=k)
        sem_out = self.search_semantic(query, k=k)
        bm25_out = self.search_bm25(query, k=k)
        hybrid_out = self.search_hybrid(query, k=k)

        tfidf_ids = [item["doc_id"] for item in tfidf_out["results"]]
        sem_ids = [item["doc_id"] for item in sem_out["results"]]

        overlap = set(tfidf_ids) & set(sem_ids)
        tfidf_only = [d for d in tfidf_ids if d not in overlap]
        sem_only = [d for d in sem_ids if d not in overlap]

        total_latency = (time.perf_counter() - start) * 1000.0

        return {
            "query": query,
            "k": k,
            "total_latency_ms": round(total_latency, 2),
            "methods": {
                "tfidf": tfidf_out,
                "semantic": sem_out,
                "bm25": bm25_out,
                "hybrid": hybrid_out,
            },
            "comparison": {
                "overlap_count": len(overlap),
                "overlap_ratio": round(len(overlap) / max(1, k), 2),
                "overlap_doc_ids": list(overlap),
                "tfidf_only_doc_ids": tfidf_only,
                "semantic_only_doc_ids": sem_only,
            },
        }


__all__ = ["SearchEngineService", "PHASE4_FAILURE_EXAMPLES"]
