"""BM25 lexical retriever wrapping BM25Okapi (rank_bm25).

Implements the standard BaseRetriever interface, supporting:
- Parameterized BM25 (k1, b, epsilon)
- Fast scoring via rank_bm25
- Top-k ranking
- Serialization to disk
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rank_bm25 import BM25Okapi

from src.lexical.base import BaseRetriever, SearchResult
from src.preprocess.tokenizer import TextPreprocessor

logger = logging.getLogger("bm25_retriever")


class BM25Retriever(BaseRetriever):
    """BM25 lexical retriever implementation based on BM25Okapi."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.preprocessor = preprocessor or TextPreprocessor()

        self.bm25: Optional[BM25Okapi] = None
        self.doc_records: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    def index(self, corpus: List[Dict[str, Any]]) -> None:
        """Build the BM25 index from corpus records."""
        if not corpus:
            raise ValueError("Corpus cannot be empty for indexing")

        self.doc_records = corpus
        tokenized_corpus: List[List[str]] = []

        for doc in corpus:
            if "tokens" in doc and isinstance(doc["tokens"], list):
                tokenized_corpus.append(doc["tokens"])
            elif "processed_text" in doc and isinstance(doc["processed_text"], str):
                tokenized_corpus.append(doc["processed_text"].split())
            elif "raw_text" in doc:
                tokenized_corpus.append(self.preprocessor.preprocess(doc["raw_text"]))
            elif "clean_body_text" in doc:
                tokenized_corpus.append(self.preprocessor.preprocess(doc["clean_body_text"]))
            else:
                tokenized_corpus.append([])

        self.tokenized_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(
            corpus=tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon,
        )
        logger.info("BM25 index built with %d documents (k1=%.2f, b=%.2f)", len(corpus), self.k1, self.b)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Score all documents against query and return top-k matches."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built yet")

        query_tokens = self.preprocessor.preprocess(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        positive_indices = np.where(scores > 0.0)[0]
        if len(positive_indices) == 0:
            return []

        pos_scores = scores[positive_indices]
        if len(positive_indices) > k:
            partition_idx = np.argpartition(pos_scores, -k)[-k:]
            top_indices = positive_indices[partition_idx]
            top_scores = scores[top_indices]
            sort_order = np.argsort(top_scores)[::-1]
            ranked_indices = top_indices[sort_order]
        else:
            sort_order = np.argsort(pos_scores)[::-1]
            ranked_indices = positive_indices[sort_order]

        results: List[Tuple[str, float]] = []
        for idx in ranked_indices[:k]:
            rec = self.doc_records[idx]
            doc_id = rec.get("chunk_id") or rec.get("doc_id", str(idx))
            results.append((doc_id, float(scores[idx])))

        return results

    def search_with_metadata(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search and format results with document metadata and text snippets."""
        raw_results = self.search(query, k=k)
        if not raw_results:
            return []

        id_to_record = {
            (r.get("chunk_id") or r.get("doc_id", str(i))): r
            for i, r in enumerate(self.doc_records)
        }

        results: List[SearchResult] = []
        for doc_id, score in raw_results:
            rec = id_to_record.get(doc_id, {})
            snippet = rec.get("raw_text") or rec.get("clean_body_text", "")
            if len(snippet) > 250:
                snippet = snippet[:250].rsplit(" ", 1)[0] + "..."

            results.append(
                SearchResult(
                    doc_id=rec.get("doc_id", doc_id),
                    chunk_id=rec.get("chunk_id", doc_id),
                    score=round(score, 5),
                    title=rec.get("title", ""),
                    url=rec.get("url", ""),
                    snippet=snippet,
                )
            )
        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persist BM25 index artifacts to disk."""
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.bm25 is None:
            raise RuntimeError("Cannot save uninitialized BM25 index")

        with open(save_path / "bm25_model.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        meta = {
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "n_docs": len(self.doc_records),
        }
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(save_path / "doc_records.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_records, f, ensure_ascii=False)

        logger.info("Saved BM25 index to %s", save_path)

    def load(self, directory: Union[str, Path]) -> None:
        """Load persisted BM25 index artifacts from disk."""
        load_path = Path(directory)

        with open(load_path / "bm25_model.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.k1 = meta["k1"]
        self.b = meta["b"]
        self.epsilon = meta["epsilon"]

        with open(load_path / "doc_records.json", "r", encoding="utf-8") as f:
            self.doc_records = json.load(f)

        logger.info("Loaded BM25 index from %s (%d docs)", load_path, len(self.doc_records))
