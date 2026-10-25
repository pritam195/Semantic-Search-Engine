"""FAISS index wrapper supporting exact (IndexFlatIP) and approximate (IndexHNSWFlat) dense search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import faiss
import numpy as np

logger = logging.getLogger("faiss_index")


class FaissIndex:
    """Wrapper managing FAISS IndexFlatIP (exact) and IndexHNSWFlat (approximate)."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "exact",
        hnsw_m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
    ) -> None:
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index: Union[faiss.IndexFlatIP, faiss.IndexHNSWFlat]
        if self.index_type == "exact":
            self.index = faiss.IndexFlatIP(dimension)
            logger.info("Initialized FAISS IndexFlatIP (Exact, dim=%d)", dimension)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
            logger.info(
                "Initialized FAISS IndexHNSWFlat (ANN, dim=%d, M=%d, efConst=%d, efSearch=%d)",
                dimension,
                hnsw_m,
                ef_construction,
                ef_search,
            )
        else:
            raise ValueError(f"Unsupported FAISS index_type: '{index_type}'. Must be 'exact' or 'hnsw'.")

    @property
    def ntotal(self) -> int:
        """Return the number of vectors indexed."""
        return self.index.ntotal

    def add(self, embeddings: np.ndarray) -> None:
        """Add dense vector embeddings to the index.

        Args:
            embeddings: 2D NumPy array of shape (N, dimension) with float32 dtype.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected embeddings shape (N, {self.dimension}), got {embeddings.shape}")

        contiguous = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(contiguous)
        logger.info("Added %d vectors to %s FAISS index (Total: %d)", len(embeddings), self.index_type, self.index.ntotal)

    def search(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k nearest neighbors for a single query vector.

        Args:
            query_vec: 1D array of shape (dimension,) or 2D array of shape (1, dimension).
            k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple of (scores_1d, indices_1d).
        """
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]
        if query_vec.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {query_vec.shape[1]}")

        q_contig = np.ascontiguousarray(query_vec, dtype=np.float32)
        top_k = min(k, self.index.ntotal)
        if top_k == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

        distances, indices = self.index.search(q_contig, top_k)
        return distances[0], indices[0]

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist FAISS index binary structure to disk."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target))
        logger.info("Saved %s FAISS index to %s (size: %.2f MB)", self.index_type, target, target.stat().st_size / (1024 * 1024))

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        index_type: str = "exact",
        ef_search: int = 64,
    ) -> FaissIndex:
        """Load a persisted FAISS index binary file."""
        target = Path(filepath)
        if not target.exists():
            raise FileNotFoundError(f"FAISS index file not found at {target}")

        raw_index = faiss.read_index(str(target))
        dim = raw_index.d

        instance = cls(dimension=dim, index_type=index_type, ef_search=ef_search)
        instance.index = raw_index
        if hasattr(instance.index, "hnsw"):
            instance.index.hnsw.efSearch = ef_search

        logger.info("Loaded %s FAISS index from %s (%d vectors, dim=%d)", index_type, target, instance.ntotal, dim)
        return instance
