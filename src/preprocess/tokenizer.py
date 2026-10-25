"""Configurable text preprocessing and tokenization pipeline.

Supports:
- Lowercasing
- Punctuation and symbol stripping
- Stopword removal (standard English or custom list)
- Optional stemming (Porter) or lemmatization (WordNet)
- Token length filtering
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional, Set, Union

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem.porter import PorterStemmer
except ImportError:
    PorterStemmer = None  # type: ignore

try:
    from nltk.stem import WordNetLemmatizer
except ImportError:
    WordNetLemmatizer = None  # type: ignore

# Matches sequences of alphanumeric characters
WORD_PATTERN = re.compile(r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)?\b")
PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)


class TextPreprocessor:
    """Configurable text cleaner, tokenizer, and normalizer."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        stopwords: Optional[Union[str, Set[str], List[str]]] = "english",
        stemmer: Optional[str] = "porter",
        min_token_length: int = 2,
        max_token_length: int = 40,
    ) -> None:
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length

        # Setup stopwords
        if stopwords == "english":
            self.stopwords: Set[str] = set(ENGLISH_STOP_WORDS)
        elif isinstance(stopwords, (set, list)):
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set()

        # Setup stemmer / lemmatizer
        self.stemmer_name = (stemmer or "none").lower()
        self._stemmer_fn = None

        if self.stemmer_name == "porter":
            if PorterStemmer is not None:
                ps = PorterStemmer()
                self._stemmer_fn = ps.stem
            else:
                self._stemmer_fn = None
        elif self.stemmer_name == "wordnet":
            if WordNetLemmatizer is not None:
                try:
                    import nltk
                    nltk.download("wordnet", quiet=True)
                    wnl = WordNetLemmatizer()
                    self._stemmer_fn = wnl.lemmatize
                except Exception:
                    self._stemmer_fn = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> TextPreprocessor:
        """Instantiate preprocessor from a dictionary configuration."""
        return cls(
            lowercase=cfg.get("lowercase", True),
            remove_punctuation=cfg.get("remove_punctuation", True),
            remove_numbers=cfg.get("remove_numbers", False),
            stopwords=cfg.get("stopwords", "english"),
            stemmer=cfg.get("stemmer", "porter"),
            min_token_length=cfg.get("min_token_length", 2),
            max_token_length=cfg.get("max_token_length", 40),
        )

    def preprocess(self, text: str) -> List[str]:
        """Transform raw text into a list of normalized tokens.

        Args:
            text: Input raw string.

        Returns:
            List of normalized string tokens.
        """
        if not text:
            return []

        if self.lowercase:
            text = text.lower()

        # Extract words matching token pattern
        tokens = WORD_PATTERN.findall(text)

        processed: List[str] = []
        for token in tokens:
            if self.remove_punctuation:
                token = token.translate(PUNCT_TRANSLATOR)

            if not token:
                continue

            if self.remove_numbers and token.isdigit():
                continue

            # Length bounds
            if len(token) < self.min_token_length or len(token) > self.max_token_length:
                continue

            # Stopword check
            if token in self.stopwords:
                continue

            # Stemming / Lemmatization
            if self._stemmer_fn is not None:
                token = self._stemmer_fn(token)

            processed.append(token)

        return processed

    def preprocess_to_str(self, text: str) -> str:
        """Preprocess text and join tokens into a space-separated string."""
        return " ".join(self.preprocess(text))
