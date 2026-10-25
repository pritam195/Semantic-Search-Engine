"""Polite, resumable, and deterministic web crawler for Wikipedia articles.

Features:
- Respects robots.txt using urllib.robotparser
- Rate-limiting (1-2 req/s) with exponential backoff and jitter
- URL normalization and deduplication
- Content-hash deduplication (SHA-256)
- Resumable state persistence to disk (frontier and visited sets)
- HTML cleaning (strips nav, infoboxes, reflists, scripts, edit sections)
- Filtering for minimum word count and language
- Appends extracted documents directly to JSONL in data/raw/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
import urllib.parse
import urllib.robotparser
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("crawler")

EXCLUDED_NAMESPACES = (
    "File:",
    "Talk:",
    "User:",
    "User_talk:",
    "Wikipedia:",
    "Wikipedia_talk:",
    "Special:",
    "Help:",
    "Help_talk:",
    "Template:",
    "Template_talk:",
    "Portal:",
    "Portal_talk:",
    "Draft:",
    "Draft_talk:",
    "TimedText:",
    "Module:",
    "MediaWiki:",
    "Category_talk:",
    "Category talk:",
)


def normalize_url(raw_url: str, base_url: str = "https://en.wikipedia.org") -> str:
    """Normalize a URL to canonical form.

    - Resolves relative paths against base_url
    - Enforces https scheme and lowercase host
    - Strips fragment identifiers and extraneous query parameters
    - Converts spaces to underscores for Wikipedia paths
    - Strips trailing slash (except for bare domain)

    Args:
        raw_url: Input URL string (absolute or relative).
        base_url: Base URL to resolve relative URLs.

    Returns:
        Canonical normalized URL string.
    """
    joined = urllib.parse.urljoin(base_url, raw_url.strip())
    parsed = urllib.parse.urlparse(joined)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Normalize Wikipedia article paths: unquote and replace spaces with underscores
    path = parsed.path
    if path.startswith("/wiki/"):
        article_title = path[len("/wiki/") :]
        # Decode safe characters, normalize spaces to underscores
        decoded = urllib.parse.unquote(article_title)
        normalized_title = decoded.strip().replace(" ", "_")
        # Re-quote while keeping colons and common characters intact
        path = "/wiki/" + urllib.parse.quote(normalized_title, safe=":/_-%()[]")

    # Remove trailing slash unless it's just "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # Drop all fragments and Wikipedia navigation query params (?action=..., ?oldid=...)
    return urllib.parse.urlunparse((scheme, netloc, path, "", "", ""))


def is_valid_article_url(url: str, allowed_domain: str = "en.wikipedia.org") -> bool:
    """Check if a URL points to an English Wikipedia article (not special namespaces).

    Args:
        url: Normalized URL.
        allowed_domain: Domain filter.

    Returns:
        True if the URL is a crawlable Wikipedia article.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != allowed_domain:
        return False
    if not parsed.path.startswith("/wiki/"):
        return False

    title = parsed.path[len("/wiki/") :]
    if not title:
        return False
    if title.lower() in ("main_page", "main_page/"):
        return False

    for ns in EXCLUDED_NAMESPACES:
        if title.startswith(ns):
            return False

    return True


def is_category_url(url: str, allowed_domain: str = "en.wikipedia.org") -> bool:
    """Check if a URL is a Wikipedia category page for link discovery."""
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != allowed_domain:
        return False
    return parsed.path.startswith("/wiki/Category:")


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text for deduplication.

    Args:
        text: Input body text.

    Returns:
        Hexadecimal SHA-256 string.
    """
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class RobotsChecker:
    """Helper to check robots.txt compliance."""

    def __init__(self, robots_url: str = "https://en.wikipedia.org/robots.txt", user_agent: str = "*") -> None:
        self.robots_url = robots_url
        self.user_agent = user_agent
        self.parser = urllib.robotparser.RobotFileParser()
        self.parser.set_url(robots_url)
        self._loaded = False

    def fetch(self, timeout: int = 10) -> bool:
        """Fetch and read robots.txt."""
        try:
            req = requests.get(self.robots_url, timeout=timeout, headers={"User-Agent": self.user_agent})
            if req.status_code == 200:
                self.parser.parse(req.text.splitlines())
                self._loaded = True
                logger.info("Successfully loaded and parsed robots.txt from %s", self.robots_url)
                return True
        except Exception as e:
            logger.warning("Failed to fetch robots.txt (%s); proceeding with polite defaults", e)
        return False

    def can_fetch(self, url: str) -> bool:
        """Return whether user_agent is allowed to fetch the given URL."""
        if not self._loaded:
            return True
        return self.parser.can_fetch(self.user_agent, url)


@dataclass
class ExtractedDocument:
    """Schema for an extracted and cleaned document."""

    doc_id: str
    url: str
    title: str
    clean_body_text: str
    headings: List[Dict[str, str]]
    crawl_timestamp: str
    word_count: int
    language: str


def extract_document(
    html: str,
    url: str,
    min_word_count: int = 100,
    allowed_languages: Optional[List[str]] = None,
) -> Tuple[Optional[ExtractedDocument], str, List[str]]:
    """Parse HTML, clean boilerplate, extract text/headings, filter and discover outlinks.

    Args:
        html: Raw HTML content string.
        url: URL of the document.
        min_word_count: Minimum word threshold.
        allowed_languages: Allowed language codes (e.g. ['en']).

    Returns:
        Tuple of (ExtractedDocument or None, status_string, list_of_discovered_urls).
    """
    if allowed_languages is None:
        allowed_languages = ["en"]

    soup = BeautifulSoup(html, "lxml")

    # 1. Language check
    html_tag = soup.find("html")
    page_lang = (html_tag.get("lang") if html_tag and html_tag.get("lang") else "en").lower()
    # Normalize language tags like 'en-US' -> 'en'
    primary_lang = page_lang.split("-")[0]
    if primary_lang not in allowed_languages:
        return None, f"filtered_lang:{primary_lang}", []

    # 2. Extract outlinks before stripping tags
    discovered_urls: List[str] = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        norm = normalize_url(href, base_url=url)
        if is_valid_article_url(norm) or is_category_url(norm):
            discovered_urls.append(norm)

    # 3. Extract title
    title = ""
    h1_tag = soup.find("h1", id="firstHeading")
    if h1_tag:
        title = h1_tag.get_text(strip=True)
    elif soup.title and soup.title.string:
        title = soup.title.string.split(" - Wikipedia")[0].strip()

    # 4. Remove boilerplate elements
    elements_to_remove = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "noscript",
        "form",
        ".infobox",
        ".navbox",
        ".reflist",
        ".printfooter",
        "#mw-navigation",
        "#siteNotice",
        ".mw-jump-link",
        "#toc",
        ".mw-editsection",
        ".hatnote",
        ".shortdescription",
        ".citation",
        ".reference",
        "sup.reference",
    ]
    for selector in elements_to_remove:
        for element in soup.select(selector):
            element.decompose()

    # 5. Extract headings
    content_root = soup.find("div", id="mw-content-text") or soup.find("body") or soup
    headings: List[Dict[str, str]] = []
    for h in content_root.find_all(["h2", "h3", "h4"]):
        h_text = h.get_text(separator=" ", strip=True)
        # Filter out Wikipedia default section headers like 'References', 'External links', 'See also'
        if h_text.lower() in ("references", "external links", "see also", "further reading", "notes"):
            continue
        if h_text:
            headings.append({"level": h.name, "text": h_text})

    # 6. Extract clean body text
    body_paragraphs: List[str] = []
    for block in content_root.find_all(["p", "ul", "ol", "dl"]):
        # Ignore references lists or nav items
        if block.find_parent(class_=["reflist", "navbox", "infobox"]):
            continue
        text = block.get_text(separator=" ", strip=True)
        if text:
            body_paragraphs.append(text)

    clean_body_text = "\n\n".join(body_paragraphs)
    words = clean_body_text.split()
    word_count = len(words)

    # 7. Word count filter
    if word_count < min_word_count:
        return None, f"filtered_short:{word_count}", discovered_urls

    doc_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    timestamp = datetime.now(timezone.utc).isoformat()

    doc = ExtractedDocument(
        doc_id=doc_id,
        url=url,
        title=title,
        clean_body_text=clean_body_text,
        headings=headings,
        crawl_timestamp=timestamp,
        word_count=word_count,
        language=primary_lang,
    )
    return doc, "ok", discovered_urls


@dataclass
class CrawlerState:
    """State storage for pausing and resuming crawler execution."""

    frontier: deque[str] = field(default_factory=deque)
    visited_urls: Set[str] = field(default_factory=set)
    content_hashes: Set[str] = field(default_factory=set)
    stats: Dict[str, int] = field(
        default_factory=lambda: {
            "total_crawled": 0,
            "total_saved": 0,
            "duplicates_url": 0,
            "duplicates_content": 0,
            "filtered_short": 0,
            "filtered_lang": 0,
            "errors": 0,
        }
    )

    def save_to_disk(self, filepath: Path) -> None:
        """Persist frontier, visited URLs, content hashes, and statistics to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_path = filepath.with_suffix(".tmp")
        data = {
            "frontier": list(self.frontier),
            "visited_urls": list(self.visited_urls),
            "content_hashes": list(self.content_hashes),
            "stats": self.stats,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, filepath)
        logger.info(
            "State saved: %d visited, %d in frontier, %d saved docs",
            len(self.visited_urls),
            len(self.frontier),
            self.stats["total_saved"],
        )

    @classmethod
    def load_from_disk(cls, filepath: Path) -> CrawlerState:
        """Load state from disk if present, else return fresh state."""
        if not filepath.exists():
            return cls()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = cls(
                frontier=deque(data.get("frontier", [])),
                visited_urls=set(data.get("visited_urls", [])),
                content_hashes=set(data.get("content_hashes", [])),
                stats=data.get("stats", {}),
            )
            logger.info(
                "Resumed crawler state from %s: %d visited, %d frontier, %d saved",
                filepath,
                len(state.visited_urls),
                len(state.frontier),
                state.stats.get("total_saved", 0),
            )
            return state
        except Exception as e:
            logger.error("Failed to load state from %s: %s; starting fresh", filepath, e)
            return cls()


class PoliteDownloader:
    """Handles HTTP requests with rate limiting and exponential retries."""

    def __init__(
        self,
        user_agent: str,
        min_delay_seconds: float = 0.6,
        timeout: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        self.user_agent = user_agent
        self.min_delay = min_delay_seconds
        self.timeout = timeout
        self.last_request_time: float = 0.0

        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": self.user_agent})

    def fetch(self, url: str) -> Optional[requests.Response]:
        """Fetch URL respecting the polite rate-limit delay."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        try:
            resp = self.session.get(url, timeout=self.timeout)
            self.last_request_time = time.time()
            return resp
        except Exception as e:
            logger.warning("Network error fetching %s: %s", url, e)
            self.last_request_time = time.time()
            return None


class WikipediaSpider:
    """Coordinates the crawl, state management, and document storage."""

    def __init__(self, config_path: str = "configs/crawler_config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.state_file = Path(self.config["storage"]["state_file"])
        self.output_jsonl = Path(self.config["storage"]["output_jsonl"])
        self.target_docs: int = self.config.get("target_docs", 5000)
        self.min_word_count: int = self.config.get("filtering", {}).get("min_word_count", 100)
        self.allowed_languages: List[str] = self.config.get("filtering", {}).get("allowed_languages", ["en"])
        self.save_every_n: int = self.config.get("storage", {}).get("save_every_n", 25)

        # Setup downloader and robots checker
        ua = self.config.get("user_agent", "SemanticSearchEngineCrawler/1.0")
        rate_cfg = self.config.get("rate_limit", {})
        net_cfg = self.config.get("network", {})
        self.downloader = PoliteDownloader(
            user_agent=ua,
            min_delay_seconds=rate_cfg.get("min_delay_seconds", 0.6),
            timeout=net_cfg.get("timeout_seconds", 10),
            max_retries=net_cfg.get("max_retries", 3),
            backoff_factor=net_cfg.get("backoff_factor", 1.5),
        )

        self.robots = RobotsChecker(user_agent=ua)
        self.robots.fetch()

        # Load or initialize state
        self.state = CrawlerState.load_from_disk(self.state_file)

        # Initialize seeds if frontier is empty
        if not self.state.frontier:
            for seed in self.config.get("seed_urls", []):
                norm_seed = normalize_url(seed)
                if norm_seed not in self.state.visited_urls:
                    self.state.frontier.append(norm_seed)

        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        logger.info("Signal %d received: saving state and shutting down gracefully...", signum)
        self.interrupted = True

    def run(self, max_docs_override: Optional[int] = None) -> None:
        """Execute the crawling loop until target_docs is reached or interrupted."""
        target = max_docs_override if max_docs_override is not None else self.target_docs
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting crawl: Target = %d docs | Saved = %d | Frontier = %d",
            target,
            self.state.stats["total_saved"],
            len(self.state.frontier),
        )

        docs_saved_this_session = 0

        with open(self.output_jsonl, "a", encoding="utf-8") as jsonl_file:
            while self.state.frontier and self.state.stats["total_saved"] < target and not self.interrupted:
                current_url = self.state.frontier.popleft()
                norm_url = normalize_url(current_url)

                if norm_url in self.state.visited_urls:
                    self.state.stats["duplicates_url"] += 1
                    continue

                self.state.visited_urls.add(norm_url)
                self.state.stats["total_crawled"] += 1

                # Check robots.txt
                if not self.robots.can_fetch(norm_url):
                    logger.info("Blocked by robots.txt: %s", norm_url)
                    continue

                resp = self.downloader.fetch(norm_url)
                if resp is None or resp.status_code != 200:
                    self.state.stats["errors"] += 1
                    continue

                # Parse and extract
                is_cat = is_category_url(norm_url)
                doc, status, discovered_links = extract_document(
                    html=resp.text,
                    url=norm_url,
                    min_word_count=self.min_word_count,
                    allowed_languages=self.allowed_languages,
                )

                # Add newly discovered links to frontier
                for link in discovered_links:
                    if link not in self.state.visited_urls and link not in self.state.frontier:
                        self.state.frontier.append(link)

                # If this was a category page, we only wanted outlinks, not to index the category listing
                if is_cat:
                    continue

                if status.startswith("filtered_short"):
                    self.state.stats["filtered_short"] += 1
                    continue
                if status.startswith("filtered_lang"):
                    self.state.stats["filtered_lang"] += 1
                    continue

                if doc is not None:
                    # Content deduplication check
                    content_hash = compute_content_hash(doc.clean_body_text)
                    if content_hash in self.state.content_hashes:
                        self.state.stats["duplicates_content"] += 1
                        logger.debug("Duplicate content hash detected for: %s", norm_url)
                        continue

                    # Persist valid document
                    self.state.content_hashes.add(content_hash)
                    json_str = json.dumps(asdict(doc), ensure_ascii=False)
                    jsonl_file.write(json_str + "\n")
                    jsonl_file.flush()

                    self.state.stats["total_saved"] += 1
                    docs_saved_this_session += 1

                    if self.state.stats["total_saved"] % 10 == 0:
                        logger.info(
                            "Progress: %d/%d docs saved (Frontier: %d | URLs seen: %d | Content dupes: %d)",
                            self.state.stats["total_saved"],
                            target,
                            len(self.state.frontier),
                            len(self.state.visited_urls),
                            self.state.stats["duplicates_content"],
                        )

                    # Periodic state checkpoint
                    if docs_saved_this_session % self.save_every_n == 0:
                        self.state.save_to_disk(self.state_file)

        # Final save upon loop exit or interrupt
        self.state.save_to_disk(self.state_file)
        logger.info(
            "Crawl finished. Saved %d documents. Total crawled: %d. Duplicates: URL=%d, Content=%d.",
            self.state.stats["total_saved"],
            self.state.stats["total_crawled"],
            self.state.stats["duplicates_url"],
            self.state.stats["duplicates_content"],
        )


def main() -> None:
    """CLI entrypoint for running the Wikipedia spider."""
    parser = argparse.ArgumentParser(description="Polite, resumable Wikipedia crawler")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/crawler_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--target-docs",
        type=int,
        default=None,
        help="Override target document count (e.g. --target-docs 50 for quick test)",
    )
    args = parser.parse_args()

    spider = WikipediaSpider(config_path=args.config)
    spider.run(max_docs_override=args.target_docs)


if __name__ == "__main__":
    main()
