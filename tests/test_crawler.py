"""Unit tests for crawler logic, URL normalization, deduplication, and HTML extraction."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

from src.crawler.spider import (
    CrawlerState,
    RobotsChecker,
    compute_content_hash,
    extract_document,
    is_category_url,
    is_valid_article_url,
    normalize_url,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestUrlNormalization:
    """Test URL canonicalization and filtering rules."""

    def test_relative_url_resolution(self) -> None:
        raw = "/wiki/Supervised_learning"
        normalized = normalize_url(raw, base_url="https://en.wikipedia.org/wiki/Machine_learning")
        assert normalized == "https://en.wikipedia.org/wiki/Supervised_learning"

    def test_fragment_stripping(self) -> None:
        raw = "https://en.wikipedia.org/wiki/Support_vector_machine#Kernel_trick"
        normalized = normalize_url(raw)
        assert normalized == "https://en.wikipedia.org/wiki/Support_vector_machine"

    def test_query_parameter_stripping(self) -> None:
        raw = "https://en.wikipedia.org/wiki/Deep_learning?action=edit&section=2"
        normalized = normalize_url(raw)
        assert normalized == "https://en.wikipedia.org/wiki/Deep_learning"

    def test_space_to_underscore_conversion(self) -> None:
        raw = "https://en.wikipedia.org/wiki/Random forest classifier"
        normalized = normalize_url(raw)
        assert normalized == "https://en.wikipedia.org/wiki/Random_forest_classifier"

    def test_trailing_slash_removal(self) -> None:
        raw = "https://en.wikipedia.org/wiki/Neural_network/"
        normalized = normalize_url(raw)
        assert normalized == "https://en.wikipedia.org/wiki/Neural_network"

    def test_lowercase_scheme_and_host(self) -> None:
        raw = "HTTPS://EN.WIKIPEDIA.ORG/wiki/Gradient_descent"
        normalized = normalize_url(raw)
        assert normalized == "https://en.wikipedia.org/wiki/Gradient_descent"

    def test_is_valid_article_url(self) -> None:
        assert is_valid_article_url("https://en.wikipedia.org/wiki/Machine_learning")
        assert is_valid_article_url("https://en.wikipedia.org/wiki/K-means_clustering")
        # Namespace and special page exclusions
        assert not is_valid_article_url("https://en.wikipedia.org/wiki/Main_Page")
        assert not is_valid_article_url("https://en.wikipedia.org/wiki/Category_talk:Machine_learning")
        assert not is_valid_article_url("https://en.wikipedia.org/wiki/Special:RecentChanges")
        assert not is_valid_article_url("https://en.wikipedia.org/wiki/File:Example.jpg")
        assert not is_valid_article_url("https://en.wikipedia.org/wiki/Help:Contents")
        assert not is_valid_article_url("https://en.wikipedia.org/wiki/Talk:Machine_learning")
        assert not is_valid_article_url("https://external.org/wiki/Machine_learning")

    def test_is_category_url(self) -> None:
        assert is_category_url("https://en.wikipedia.org/wiki/Category:Machine_learning")
        assert not is_category_url("https://en.wikipedia.org/wiki/Machine_learning")


class TestContentHashing:
    """Test content hash generation and invariance."""

    def test_whitespace_and_casing_invariance(self) -> None:
        text1 = "Machine learning algorithms build a mathematical model based on sample data."
        text2 = "  Machine   learning algorithms  build a mathematical model\nbased on sample data.  "
        assert compute_content_hash(text1) == compute_content_hash(text2)

    def test_distinct_content_different_hash(self) -> None:
        text1 = "Gradient descent is a first-order iterative optimization algorithm."
        text2 = "Newton's method is a second-order optimization method using Hessian matrices."
        assert compute_content_hash(text1) != compute_content_hash(text2)


class TestHtmlExtraction:
    """Test text cleaning, boilerplate stripping, and metadata extraction on fixtures."""

    @pytest.fixture
    def sample_wikipedia_html(self) -> str:
        fixture_path = FIXTURES_DIR / "sample_wikipedia_page.html"
        return fixture_path.read_text(encoding="utf-8")

    @pytest.fixture
    def sample_short_html(self) -> str:
        fixture_path = FIXTURES_DIR / "sample_short_page.html"
        return fixture_path.read_text(encoding="utf-8")

    @pytest.fixture
    def sample_foreign_html(self) -> str:
        fixture_path = FIXTURES_DIR / "sample_foreign_page.html"
        return fixture_path.read_text(encoding="utf-8")

    def test_boilerplate_stripping(self, sample_wikipedia_html: str) -> None:
        url = "https://en.wikipedia.org/wiki/Machine_Learning"
        doc, status, outlinks = extract_document(sample_wikipedia_html, url, min_word_count=100)

        assert status == "ok"
        assert doc is not None
        assert doc.title == "Machine Learning"

        # Ensure boilerplate elements are stripped
        assert "var x = 123" not in doc.clean_body_text
        assert "Navigation box at bottom" not in doc.clean_body_text
        assert "Infobox details" not in doc.clean_body_text
        assert "Table of contents" not in doc.clean_body_text
        assert "Citations and references to strip" not in doc.clean_body_text
        assert "[edit]" not in doc.clean_body_text

        # Ensure clean content is present
        assert "Machine learning (ML) is a field of inquiry" in doc.clean_body_text
        assert "Supervised learning algorithms build a mathematical model" in doc.clean_body_text

    def test_headings_extraction(self, sample_wikipedia_html: str) -> None:
        url = "https://en.wikipedia.org/wiki/Machine_Learning"
        doc, status, _ = extract_document(sample_wikipedia_html, url, min_word_count=100)

        assert doc is not None
        heading_texts = [h["text"] for h in doc.headings]
        assert "Overview" in heading_texts
        assert "Supervised Learning" in heading_texts
        assert any(h["level"] == "h2" and "Overview" in h["text"] for h in doc.headings)
        assert any(h["level"] == "h3" and "Supervised Learning" in h["text"] for h in doc.headings)

    def test_short_document_filter(self, sample_short_html: str) -> None:
        url = "https://en.wikipedia.org/wiki/Short_Page"
        doc, status, _ = extract_document(sample_short_html, url, min_word_count=100)

        assert doc is None
        assert status.startswith("filtered_short")

    def test_foreign_language_filter(self, sample_foreign_html: str) -> None:
        url = "https://en.wikipedia.org/wiki/Maschinelles_Lernen"
        doc, status, _ = extract_document(sample_foreign_html, url, allowed_languages=["en"])

        assert doc is None
        assert status == "filtered_lang:de"


class TestCrawlerStatePersistence:
    """Test state saving and resumption from disk."""

    def test_save_and_load_state(self, tmp_path: Path) -> None:
        state_file = tmp_path / "test_crawler_state.json"

        state = CrawlerState(
            frontier=deque(["https://en.wikipedia.org/wiki/Page_A", "https://en.wikipedia.org/wiki/Page_B"]),
            visited_urls={"https://en.wikipedia.org/wiki/Page_C"},
            content_hashes={"abc123hash"},
            stats={"total_crawled": 5, "total_saved": 1, "duplicates_url": 2, "duplicates_content": 1, "filtered_short": 1, "filtered_lang": 0, "errors": 0},
        )
        state.save_to_disk(state_file)

        assert state_file.exists()
        loaded = CrawlerState.load_from_disk(state_file)

        assert list(loaded.frontier) == ["https://en.wikipedia.org/wiki/Page_A", "https://en.wikipedia.org/wiki/Page_B"]
        assert loaded.visited_urls == {"https://en.wikipedia.org/wiki/Page_C"}
        assert loaded.content_hashes == {"abc123hash"}
        assert loaded.stats["total_saved"] == 1
        assert loaded.stats["duplicates_url"] == 2


class TestRobotsCompliance:
    """Test robots.txt compliance logic."""

    def test_robots_allowed_and_disallowed(self) -> None:
        checker = RobotsChecker()
        checker._loaded = True
        checker.parser.parse(
            """
User-agent: *
Disallow: /wiki/Special:
Disallow: /w/
Allow: /wiki/
""".splitlines()
        )

        assert checker.can_fetch("https://en.wikipedia.org/wiki/Machine_learning")
        assert not checker.can_fetch("https://en.wikipedia.org/wiki/Special:RecentChanges")
