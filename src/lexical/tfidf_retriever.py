"""TF-IDF Retriever implemented from scratch in NumPy and SciPy sparse matrices.

Adheres strictly to the course requirements:
- No use of sklearn's TfidfVectorizer for the core retrieval implementation
- Inverted index representation
- Smoothed Inverse Document Frequency (IDF)
- Sublinear Term Frequency option (1 + ln(tf))
- L2 document and query vector normalization
- Cosine similarity ranking via sparse matrix-vector dot products
- Conforms to BaseRetriever interface
"""

from __future__ import annotations

import collections
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

from src.lexical.base import BaseRetriever, SearchResult
from src.preprocess.tokenizer import TextPreprocessor

logger = logging.getLogger("tfidf_retriever")


class CustomTFIDFRetriever(BaseRetriever):
    """From-scratch TF-IDF indexer and retriever using NumPy & SciPy CSR sparse matrix."""

    def __init__(
        self,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
        norm: str = "l2",
        min_df: int = 1,
        max_df: float = 1.0,
        max_features: Optional[int] = None,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf
        self.norm = norm
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.preprocessor = preprocessor or TextPreprocessor()

        # Vocabulary and Inverted Index State
        self.vocabulary: Dict[str, int] = {}  # term -> feature_index
        self.feature_names: List[str] = []   # feature_index -> term
        self.idf_: Optional[np.ndarray] = None
        self.doc_matrix: Optional[sp.csr_matrix] = None  # (N_docs, V_vocab)
        self.doc_records: List[Dict[str, Any]] = []      # list of document metadata
        self.inverted_index: Dict[int, List[int]] = collections.defaultdict(list)  # term_idx -> list of doc_indices
        self.n_docs: int = 0

    def _build_vocabulary(self, tokenized_docs: Sequence[List[str]]) -> None:
        """Construct vocabulary filtered by min_df, max_df, and max_features."""
        n_docs = len(tokenized_docs)
        self.n_docs = n_docs
        df_counter: collections.Counter[str] = collections.Counter()
        tf_counter: collections.Counter[str] = collections.Counter()

        for doc_tokens in tokenized_docs:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                df_counter[token] += 1
            for token in doc_tokens:
                tf_counter[token] += 1

        # Apply min_df and max_df thresholds
        max_doc_count = n_docs if self.max_df == 1.0 else int(self.max_df * n_docs)
        min_doc_count = self.min_df

        candidates: List[str] = [
            term
            for term, df in df_counter.items()
            if min_doc_count <= df <= max_doc_count
        ]

        # Sort candidates for deterministic vocabulary ordering
        if self.max_features is not None and len(candidates) > self.max_features:
            # Keep top features by global term frequency, tie-break by term name
            candidates.sort(key=lambda t: (-tf_counter[t], t))
            candidates = candidates[: self.max_features]

        # Lexicographical sort matching standard IR conventions
        candidates.sort()

        self.vocabulary = {term: idx for idx, term in enumerate(candidates)}
        self.feature_names = candidates
        vocab_size = len(self.vocabulary)
        logger.info("Built vocabulary of %d terms from %d documents", vocab_size, n_docs)

    def _compute_idf(self, tokenized_docs: Sequence[List[str]]) -> np.ndarray:
        """Compute smoothed or unsmoothed Inverse Document Frequency."""
        n_docs = self.n_docs
        vocab_size = len(self.vocabulary)
        df_array = np.zeros(vocab_size, dtype=np.float64)

        for doc_tokens in tokenized_docs:
            seen_indices: Set[int] = set()
            for token in doc_tokens:
                idx = self.vocabulary.get(token)
                if idx is not None and idx not in seen_indices:
                    df_array[idx] += 1
                    seen_indices.add(idx)

        # Smoothed IDF: ln((1 + N) / (1 + df)) + 1.0
        # Unsmoothed IDF: ln(N / df) + 1.0
        if self.smooth_idf:
            idf = np.log((1.0 + n_docs) / (1.0 + df_array)) + 1.0
        else:
            idf = np.log(float(n_docs) / np.maximum(df_array, 1.0)) + 1.0

        return idf

    def index(self, corpus: List[Dict[str, Any]]) -> None:
        """Index a corpus of document or chunk records."""
        if not corpus:
            raise ValueError("Corpus cannot be empty for indexing")

        self.doc_records = corpus
        n_docs = len(corpus)

        # 1. Tokenize if not already tokenized
        tokenized_docs: List[List[str]] = []
        for doc in corpus:
            if "tokens" in doc and isinstance(doc["tokens"], list):
                tokenized_docs.append(doc["tokens"])
            elif "processed_text" in doc and isinstance(doc["processed_text"], str):
                tokenized_docs.append(doc["processed_text"].split())
            elif "raw_text" in doc:
                tokenized_docs.append(self.preprocessor.preprocess(doc["raw_text"]))
            elif "clean_body_text" in doc:
                tokenized_docs.append(self.preprocessor.preprocess(doc["clean_body_text"]))
            else:
                tokenized_docs.append([])

        # 2. Vocabulary & IDF
        self._build_vocabulary(tokenized_docs)
        vocab_size = len(self.vocabulary)
        self.idf_ = self._compute_idf(tokenized_docs)

        # 3. Construct sparse TF matrix and inverted index
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        self.inverted_index = collections.defaultdict(list)

        for doc_idx, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            term_counts: collections.Counter[int] = collections.Counter()
            for t in tokens:
                idx = self.vocabulary.get(t)
                if idx is not None:
                    term_counts[idx] += 1

            for term_idx, count in term_counts.items():
                # Sublinear TF: 1 + ln(count) if count > 0 else 0
                tf = 1.0 + math.log(count) if self.sublinear_tf else float(count)
                # Multiply by IDF
                tfidf = tf * self.idf_[term_idx]

                rows.append(doc_idx)
                cols.append(term_idx)
                data.append(tfidf)

                # Inverted index posting
                self.inverted_index[term_idx].append(doc_idx)

        # Build SciPy CSR matrix (docs x vocabulary)
        matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, vocab_size), dtype=np.float64)

        # 4. L2 Normalization across rows
        if self.norm == "l2":
            row_norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
            row_norms[row_norms == 0.0] = 1.0
            inv_norms = 1.0 / row_norms
            # Multiply rows by inv_norms
            matrix = matrix.multiply(inv_norms[:, np.newaxis]).tocsr()

        self.doc_matrix = matrix
        logger.info(
            "Custom TF-IDF index complete: %d docs, %d terms, %d non-zero entries (%.3f%% density)",
            n_docs,
            vocab_size,
            matrix.nnz,
            (matrix.nnz / max(1, n_docs * vocab_size)) * 100,
        )

    def vectorize_query(self, query: str) -> np.ndarray:
        """Transform a raw query string into an L2-normalized TF-IDF vector."""
        if self.idf_ is None:
            raise RuntimeError("Retriever has not been indexed yet")

        tokens = self.preprocessor.preprocess(query)
        vocab_size = len(self.vocabulary)
        q_vec = np.zeros(vocab_size, dtype=np.float64)

        term_counts: collections.Counter[int] = collections.Counter()
        for t in tokens:
            idx = self.vocabulary.get(t)
            if idx is not None:
                term_counts[idx] += 1

        for term_idx, count in term_counts.items():
            tf = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            q_vec[term_idx] = tf * self.idf_[term_idx]

        if self.norm == "l2":
            norm = np.linalg.norm(q_vec)
            if norm > 0.0:
                q_vec /= norm

        return q_vec

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Compute cosine similarity ranking and return top-k matches."""
        if self.doc_matrix is None or self.idf_ is None:
            raise RuntimeError("Index not built. Call .index() or .load() first.")

        q_vec = self.vectorize_query(query)
        if np.all(q_vec == 0.0):
            return []

        # Sparse matrix-vector product computes cosine similarity for all documents:
        # scores = DocMatrix @ q_vec
        scores = self.doc_matrix.dot(q_vec)

        # Filter strictly positive matches
        positive_indices = np.where(scores > 0.0)[0]
        if len(positive_indices) == 0:
            return []

        # Find top k indices
        pos_scores = scores[positive_indices]
        if len(positive_indices) > k:
            # Use argpartition for O(N) top-k selection
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
        """Persist index matrix, vocabulary, IDF weights, and metadata to disk."""
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.doc_matrix is None or self.idf_ is None:
            raise RuntimeError("Cannot save uninitialized index")

        # Save sparse matrix in SciPy .npz format
        sp.save_npz(save_path / "tfidf_matrix.npz", self.doc_matrix)

        # Save IDF array
        np.save(save_path / "idf_weights.npy", self.idf_)

        # Save vocabulary, parameters, and metadata
        meta = {
            "sublinear_tf": self.sublinear_tf,
            "smooth_idf": self.smooth_idf,
            "norm": self.norm,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "vocabulary": self.vocabulary,
            "feature_names": self.feature_names,
            "n_docs": self.n_docs,
        }
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(save_path / "doc_records.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_records, f, ensure_ascii=False)

        logger.info("Saved TF-IDF index artifacts to %s", save_path)

    def load(self, directory: Union[str, Path]) -> None:
        """Load persisted index artifacts from disk."""
        load_path = Path(directory)

        with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.sublinear_tf = meta["sublinear_tf"]
        self.smooth_idf = meta["smooth_idf"]
        self.norm = meta["norm"]
        self.min_df = meta["min_df"]
        self.max_df = meta["max_df"]
        self.max_features = meta.get("max_features")
        self.vocabulary = meta["vocabulary"]
        self.feature_names = meta["feature_names"]
        self.n_docs = meta["n_docs"]

        self.idf_ = np.load(load_path / "idf_weights.npy")
        self.doc_matrix = sp.load_npz(load_path / "tfidf_matrix.npz")

        with open(load_path / "doc_records.json", "r", encoding="utf-8") as f:
            self.doc_records = json.load(f)

        logger.info(
            "Loaded TF-IDF index from %s: %d docs, %d vocabulary features",
            load_path,
            self.n_docs,
            len(self.vocabulary),
        )
