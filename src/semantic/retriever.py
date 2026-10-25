"""Semantic Retriever using Sentence-Transformers and FAISS (Exact & ANN).

Features:
- Configurable sliding-window passage chunking (e.g. 200 words, 40 overlap)
- Chunk-to-document mapping and configurable score aggregation (max, mean, top_k_sum)
- Dual FAISS indexes: Exact (IndexFlatIP) and Approximate (IndexHNSWFlat)
- Vector L2 normalization and Inner Product Cosine scoring
- Full conformance to the Phase 2 BaseRetriever interface
"""

from __future__ import annotations

import collections
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.lexical.base import BaseRetriever, SearchResult
from src.preprocess.chunker import TextChunker
from src.semantic.embedder import SentenceEmbedder
from src.semantic.faiss_index import FaissIndex

logger = logging.getLogger("semantic_retriever")


class SemanticRetriever(BaseRetriever):
    """Dense vector retriever supporting both exact and HNSW FAISS indices."""

    def __init__(
        self,
        embedder: Optional[SentenceEmbedder] = None,
        aggregation_strategy: str = "max",
        chunk_size_words: int = 200,
        chunk_overlap_words: int = 40,
        min_chunk_words: int = 25,
        hnsw_m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
    ) -> None:
        self.embedder = embedder or SentenceEmbedder()
        self.aggregation_strategy = aggregation_strategy.lower()
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        if chunk_overlap_words >= chunk_size_words:
            chunk_overlap_words = max(0, chunk_size_words // 4)

        self.chunker = TextChunker(
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
            min_chunk_words=min_chunk_words,
        )

        self.exact_index: Optional[FaissIndex] = None
        self.ann_index: Optional[FaissIndex] = None

        self.chunk_records: List[Dict[str, Any]] = []
        self.chunk_to_doc_id: List[str] = []
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}

    def index(
        self,
        corpus: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Process corpus, generate passage embeddings, and build FAISS indices.

        Args:
            corpus: List of documents with 'doc_id', 'title', 'clean_body_text'.
            embeddings: Optional pre-computed embeddings array to skip encoding.
            batch_size: Batch size for embedding model.

        Returns:
            Computed or provided embeddings array (N_chunks, dimension).
        """
        logger.info("Chunking and preparing %d corpus documents for semantic indexing...", len(corpus))
        chunk_records: List[Dict[str, Any]] = []
        chunk_to_doc: List[str] = []
        doc_metadata: Dict[str, Dict[str, Any]] = {}

        for doc in corpus:
            doc_id = doc.get("doc_id", "")
            title = doc.get("title", "")
            url = doc.get("url", "")
            body = doc.get("clean_body_text") or doc.get("raw_text", "")

            doc_metadata[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "raw_text": body,
            }

            chunks = self.chunker.chunk_document(doc_id=doc_id, title=title, url=url, text=body)
            for c in chunks:
                c_dict = c.to_dict()
                chunk_records.append(c_dict)
                chunk_to_doc.append(doc_id)

        self.chunk_records = chunk_records
        self.chunk_to_doc_id = chunk_to_doc
        self.doc_metadata = doc_metadata
        n_chunks = len(chunk_records)
        logger.info("Produced %d chunks from %d documents.", n_chunks, len(corpus))

        # 2. Compute or reuse embeddings
        if embeddings is None:
            logger.info("Encoding %d passages with %s...", n_chunks, self.embedder.model_name)
            texts = [c["text"] for c in chunk_records]
            embeddings = self.embedder.embed_passages(texts, batch_size=batch_size)
        else:
            if len(embeddings) != n_chunks:
                raise ValueError(f"Provided embeddings count ({len(embeddings)}) != chunk count ({n_chunks})")

        dim = embeddings.shape[1]

        # 3. Build Exact Index (IndexFlatIP)
        self.exact_index = FaissIndex(dimension=dim, index_type="exact")
        self.exact_index.add(embeddings)

        # 4. Build Approximate Index (IndexHNSWFlat)
        self.ann_index = FaissIndex(
            dimension=dim,
            index_type="hnsw",
            hnsw_m=self.hnsw_m,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
        )
        self.ann_index.add(embeddings)

        return embeddings

    def _aggregate_scores(
        self,
        chunk_scores: np.ndarray,
        chunk_indices: np.ndarray,
        k: int,
    ) -> List[Tuple[str, float]]:
        """Aggregate chunk-level similarity scores to document-level scores."""
        doc_grouped: Dict[str, List[float]] = collections.defaultdict(list)

        for score, c_idx in zip(chunk_scores, chunk_indices):
            if c_idx < 0 or c_idx >= len(self.chunk_to_doc_id):
                continue
            doc_id = self.chunk_to_doc_id[c_idx]
            doc_grouped[doc_id].append(float(score))

        aggregated: List[Tuple[str, float]] = []
        for doc_id, scores_list in doc_grouped.items():
            if self.aggregation_strategy == "max":
                agg_score = max(scores_list)
            elif self.aggregation_strategy == "mean":
                agg_score = sum(scores_list) / len(scores_list)
            elif self.aggregation_strategy == "top_k_sum":
                agg_score = sum(sorted(scores_list, reverse=True)[:2])
            else:
                agg_score = max(scores_list)

            aggregated.append((doc_id, agg_score))

        # Sort documents descending by aggregated score
        aggregated.sort(key=lambda x: x[1], reverse=True)
        return aggregated[:k]

    def search(
        self,
        query: str,
        k: int = 10,
        use_ann: bool = False,
    ) -> List[Tuple[str, float]]:
        """Search dense index and return top-k aggregated (doc_id, score) pairs."""
        target_index = self.ann_index if use_ann else self.exact_index
        if target_index is None:
            raise RuntimeError("Index not built. Call index() or load() first.")

        q_vec = self.embedder.embed_query(query)
        # Retrieve extra chunks to allow sufficient document aggregation
        fetch_k = min(max(k * 8, 40), target_index.ntotal)
        scores, indices = target_index.search(q_vec, k=fetch_k)

        return self._aggregate_scores(scores, indices, k=k)

    def search_chunks(
        self,
        query: str,
        k: int = 10,
        use_ann: bool = False,
    ) -> List[Tuple[str, float]]:
        """Retrieve raw passage-level (chunk_id, score) pairs without document aggregation."""
        target_index = self.ann_index if use_ann else self.exact_index
        if target_index is None:
            raise RuntimeError("Index not built.")

        q_vec = self.embedder.embed_query(query)
        scores, indices = target_index.search(q_vec, k=k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores, indices):
            if idx >= 0 and idx < len(self.chunk_records):
                chunk_id = self.chunk_records[idx]["chunk_id"]
                results.append((chunk_id, float(score)))
        return results

    def search_with_metadata(
        self,
        query: str,
        k: int = 10,
        use_ann: bool = False,
    ) -> List[SearchResult]:
        """Search and format results with document metadata and best matching passage snippet."""
        doc_results = self.search(query, k=k, use_ann=use_ann)
        if not doc_results:
            return []

        # Find best chunk for snippet display
        q_vec = self.embedder.embed_query(query)
        target_index = self.ann_index if use_ann else self.exact_index
        assert target_index is not None
        chunk_scores, chunk_indices = target_index.search(q_vec, k=min(k * 10, target_index.ntotal))

        best_chunk_per_doc: Dict[str, Dict[str, Any]] = {}
        for s, idx in zip(chunk_scores, chunk_indices):
            if idx >= 0 and idx < len(self.chunk_records):
                rec = self.chunk_records[idx]
                d_id = rec["doc_id"]
                if d_id not in best_chunk_per_doc:
                    best_chunk_per_doc[d_id] = rec

        results: List[SearchResult] = []
        for doc_id, score in doc_results:
            meta = self.doc_metadata.get(doc_id, {})
            best_chunk = best_chunk_per_doc.get(doc_id, {})
            snippet = best_chunk.get("text") or meta.get("raw_text", "")
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit(" ", 1)[0] + "..."

            results.append(
                SearchResult(
                    doc_id=doc_id,
                    chunk_id=best_chunk.get("chunk_id", f"{doc_id}#0"),
                    score=round(score, 5),
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    snippet=snippet,
                )
            )
        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persist FAISS indices, chunk metadata, and mappings to disk."""
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.exact_index:
            self.exact_index.save(save_path / "exact.faiss")
        if self.ann_index:
            self.ann_index.save(save_path / "ann.faiss")

        meta = {
            "model_name": self.embedder.model_name,
            "dimension": self.embedder.dimension,
            "aggregation_strategy": self.aggregation_strategy,
            "chunk_size_words": self.chunker.chunk_size,
            "chunk_overlap_words": self.chunker.chunk_overlap,
            "hnsw_m": self.hnsw_m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "num_chunks": len(self.chunk_records),
            "num_docs": len(self.doc_metadata),
        }
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(save_path / "chunk_records.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_records, f, ensure_ascii=False)

        with open(save_path / "doc_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_metadata, f, ensure_ascii=False)

        with open(save_path / "chunk_to_doc.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_to_doc_id, f)

        logger.info("Saved SemanticRetriever state and FAISS indices to %s", save_path)

    def load(self, directory: Union[str, Path]) -> None:
        """Load persisted FAISS indices and metadata from disk."""
        load_path = Path(directory)

        with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.aggregation_strategy = meta["aggregation_strategy"]
        self.hnsw_m = meta.get("hnsw_m", 32)
        self.ef_construction = meta.get("ef_construction", 64)
        self.ef_search = meta.get("ef_search", 64)

        if self.embedder.model_name != meta["model_name"]:
            self.embedder = SentenceEmbedder(model_name=meta["model_name"])

        with open(load_path / "chunk_records.json", "r", encoding="utf-8") as f:
            self.chunk_records = json.load(f)

        with open(load_path / "doc_metadata.json", "r", encoding="utf-8") as f:
            self.doc_metadata = json.load(f)

        with open(load_path / "chunk_to_doc.json", "r", encoding="utf-8") as f:
            self.chunk_to_doc_id = json.load(f)

        exact_path = load_path / "exact.faiss"
        if exact_path.exists():
            self.exact_index = FaissIndex.load(exact_path, index_type="exact")

        ann_path = load_path / "ann.faiss"
        if ann_path.exists():
            self.ann_index = FaissIndex.load(ann_path, index_type="hnsw", ef_search=self.ef_search)

        logger.info("Loaded SemanticRetriever from %s (%d chunks, %d docs)", load_path, len(self.chunk_records), len(self.doc_metadata))
