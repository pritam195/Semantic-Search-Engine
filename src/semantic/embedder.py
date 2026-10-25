"""Sentence-Transformers dense embedder wrapping ONNX Runtime via fastembed.

Features:
- Supports all-MiniLM-L6-v2 and BAAI/bge-small-en-v1.5
- Automatic BGE query prefixing for retrieval
- Batch encoding with progress tracking
- Vector L2 normalization for inner product cosine equivalency
- Disk caching in NumPy .npy format
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger("embedder")

DEFAULT_BGE_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceEmbedder:
    """Dense sentence embedder supporting all-MiniLM-L6-v2 and BAAI/bge-small-en-v1.5."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        bge_query_prefix: str = DEFAULT_BGE_PREFIX,
        batch_size: int = 128,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.bge_query_prefix = bge_query_prefix
        self.batch_size = batch_size
        self.normalize = normalize_embeddings

        self.is_bge = "bge" in model_name.lower()
        logger.info("Initializing SentenceEmbedder with model: %s (is_bge=%s)", model_name, self.is_bge)

        self._model = TextEmbedding(model_name=model_name)
        # Determine embedding dimension from a probe vector
        probe = list(self._model.embed(["probe"]))[0]
        self.dimension: int = len(probe)
        logger.info("SentenceEmbedder initialized successfully. Embedding dimension = %d", self.dimension)

    def embed_passages(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Batch encode a sequence of text passages into normalized dense vectors.

        Args:
            texts: List or sequence of passage strings.
            batch_size: Batch size override.

        Returns:
            2D NumPy array of shape (N, dimension) with float32 dtype.
        """
        bs = batch_size or self.batch_size
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # FastEmbed handles streaming batch execution
        generator = self._model.embed(texts, batch_size=bs)
        embeddings_list: List[np.ndarray] = [vec for vec in generator]
        arr = np.array(embeddings_list, dtype=np.float32)

        if self.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            arr = arr / norms

        return arr

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single search query, prepending BGE prefix if applicable.

        Args:
            query: Input natural language query.

        Returns:
            1D NumPy array of shape (dimension,) with float32 dtype.
        """
        formatted_query = query
        if self.is_bge and not query.startswith(self.bge_query_prefix):
            formatted_query = f"{self.bge_query_prefix}{query}"

        # FastEmbed provides query_embed for BGE models or fallback to embed
        generator = self._model.query_embed(formatted_query) if hasattr(self._model, "query_embed") else self._model.embed([formatted_query])
        vec = list(generator)[0].astype(np.float32)

        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0.0:
                vec = vec / norm

        return vec

    @staticmethod
    def save_cache(embeddings: np.ndarray, cache_path: Union[str, Path]) -> None:
        """Persist embeddings array to disk as a binary .npy file."""
        target = Path(cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, embeddings)
        logger.info("Saved %d embeddings (%.2f MB) to %s", len(embeddings), target.stat().st_size / (1024 * 1024), target)

    @staticmethod
    def load_cache(cache_path: Union[str, Path]) -> np.ndarray:
        """Load embeddings array from a binary .npy file."""
        target = Path(cache_path)
        if not target.exists():
            raise FileNotFoundError(f"Embedding cache not found at {target}")
        arr = np.load(target).astype(np.float32)
        logger.info("Loaded %d embeddings from cache %s (dim=%d)", len(arr), target, arr.shape[1])
        return arr
