"""Abstract Base Class for lexical and semantic retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


@dataclass
class SearchResult:
    """Rich retrieval result containing doc_id, score, and passage metadata."""

    doc_id: str
    score: float
    title: str = ""
    url: str = ""
    snippet: str = ""
    chunk_id: str = ""


class BaseRetriever(ABC):
    """Abstract base retriever interface providing a uniform search API."""

    @abstractmethod
    def index(self, corpus: List[Dict[str, Any]]) -> None:
        """Build the index from a list of corpus records.

        Each record must contain at least 'chunk_id' or 'doc_id', and 'text' or 'tokens' / 'processed_text'.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search the corpus and return top-k matches as a list of (doc_id, score) pairs.

        Args:
            query: Input natural language query.
            k: Number of top results to return.

        Returns:
            List of (doc_id, score) tuples sorted in descending order of score.
        """
        pass

    @abstractmethod
    def search_with_metadata(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search and return top-k matches with title, URL, and snippet metadata."""
        pass

    @abstractmethod
    def save(self, directory: Union[str, Path]) -> None:
        """Persist index structures and vocabulary to disk."""
        pass

    @abstractmethod
    def load(self, directory: Union[str, Path]) -> None:
        """Load index structures and vocabulary from disk."""
        pass
