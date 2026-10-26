"""Cross-Encoder Reranker using ms-marco-MiniLM-L-6-v2 via ONNX / FastEmbed.

Reranks candidate document passages retrieved by first-stage retrievers
by computing full query-document cross-attention logits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("cross_encoder_reranker")

DEFAULT_CROSS_ENCODER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Neural cross-encoder reranker wrapping ms-marco-MiniLM-L-6-v2."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            logger.info("Initializing CrossEncoder with %s...", self.model_name)
            self._model = TextCrossEncoder(model_name=self.model_name)
            logger.info("CrossEncoder initialized successfully.")
        except Exception as e:
            logger.warning("Failed to initialize FastEmbed TextCrossEncoder: %s. Reranking will fallback to passthrough.", e)
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank a list of candidate documents.

        Args:
            query: User search query string.
            candidates: Sequence of (doc_id, text_snippet) tuples.
            top_k: Optional number of top reranked items to return.

        Returns:
            List of (doc_id, rerank_score) sorted descending by relevance score.
        """
        if not candidates:
            return []

        if self._model is None:
            # Fallback passthrough with dummy ranks
            return [(doc_id, float(len(candidates) - i)) for i, (doc_id, _) in enumerate(candidates)][:top_k]

        doc_ids = [doc_id for doc_id, _ in candidates]
        # Truncate text to avoid excessive compute / token limits
        texts = [text[:1000] if text else "" for _, text in candidates]

        try:
            scores_generator = self._model.rerank(query, texts)
            scores = [float(s) for s in scores_generator]
        except Exception as e:
            logger.error("Cross-encoder inference failed: %s. Using default ordering.", e)
            return [(doc_id, float(len(candidates) - i)) for i, (doc_id, _) in enumerate(candidates)][:top_k]

        reranked = list(zip(doc_ids, scores))
        reranked.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked

    def rerank_doc_lookup(
        self,
        query: str,
        candidate_ids: Sequence[str],
        doc_lookup: Dict[str, Dict[str, Any]],
        top_k: Optional[int] = None,
        text_char_limit: int = 800,
    ) -> List[Tuple[str, float]]:
        """Convenience method to rerank document IDs using a corpus document lookup."""
        candidates: List[Tuple[str, str]] = []
        for doc_id in candidate_ids:
            doc = doc_lookup.get(doc_id, {})
            title = doc.get("title", "")
            body = doc.get("clean_body_text") or doc.get("raw_text", "")
            text = f"{title}. {body[:text_char_limit]}"
            candidates.append((doc_id, text))

        return self.rerank(query, candidates, top_k=top_k)
