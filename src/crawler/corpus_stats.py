"""Corpus analysis script for generating reproducible statistics and plots.

Computes:
- Total document count
- Word length distribution (mean, median, min, max, std, quartiles)
- Histogram figure saved to reports/figures/corpus_length_hist.png
- Vocabulary size (unique lowercased alpha tokens)
- Top 25 most frequent terms (excluding common English stopwords)
- Duplicate rate (from crawler state and corpus content hashes)
- Reproducible JSON artifact saved to reports/results/corpus_stats.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("corpus_stats")

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]{2,}\b")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alpha words with at least 2 characters."""
    return TOKEN_PATTERN.findall(text.lower())


def compute_statistics(
    jsonl_path: Path,
    state_path: Path,
    figure_output_path: Path,
    results_json_path: Path,
    top_n: int = 25,
) -> Dict[str, Any]:
    """Compute and persist comprehensive corpus statistics.

    Args:
        jsonl_path: Path to documents.jsonl.
        state_path: Path to crawler_state.json.
        figure_output_path: Output PNG path for the word count histogram.
        results_json_path: Output JSON path for structured results.
        top_n: Number of top terms to return.

    Returns:
        Dictionary containing all computed corpus metrics.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Corpus file not found at {jsonl_path}")

    doc_count = 0
    word_counts: List[int] = []
    vocab_counter: collections.Counter[str] = collections.Counter()
    content_hashes: set[str] = set()
    internal_duplicates = 0

    logger.info("Reading corpus from %s...", jsonl_path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_count += 1

            body_text = doc.get("clean_body_text", "")
            words = body_text.split()
            w_len = len(words)
            word_counts.append(w_len)

            # Tokenize for vocabulary analysis
            tokens = tokenize(body_text)
            for token in tokens:
                vocab_counter[token] += 1

            # Check internal duplicate content
            doc_id = doc.get("doc_id", "")
            if doc_id in content_hashes:
                internal_duplicates += 1
            else:
                content_hashes.add(doc_id)

    if doc_count == 0:
        logger.warning("Corpus is empty.")
        return {"document_count": 0}

    # Length statistics
    word_counts.sort()
    total_words = sum(word_counts)
    mean_length = total_words / doc_count
    median_length = word_counts[doc_count // 2]
    min_length = word_counts[0]
    max_length = word_counts[-1]
    q25 = word_counts[int(0.25 * doc_count)]
    q75 = word_counts[int(0.75 * doc_count)]
    std_length = math.sqrt(sum((x - mean_length) ** 2 for x in word_counts) / doc_count)

    # Vocabulary & Top terms
    vocab_size = len(vocab_counter)
    filtered_terms = [
        (term, count)
        for term, count in vocab_counter.most_common()
        if term not in ENGLISH_STOP_WORDS and len(term) > 2
    ]
    top_terms = filtered_terms[:top_n]

    # Duplicate rate calculation from crawler state
    crawler_stats: Dict[str, Any] = {}
    total_requests = 0
    duplicate_rate = 0.0
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as sf:
                state_data = json.load(sf)
                crawler_stats = state_data.get("stats", {})
                url_dupes = crawler_stats.get("duplicates_url", 0)
                content_dupes = crawler_stats.get("duplicates_content", 0)
                crawled = crawler_stats.get("total_crawled", 0)
                total_attempts = crawled + url_dupes
                if total_attempts > 0:
                    duplicate_rate = (url_dupes + content_dupes) / total_attempts
        except Exception as e:
            logger.warning("Could not read crawler state for duplicate rate: %s", e)

    # Generate and save histogram
    figure_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=150)
    plt.hist(word_counts, bins=35, color="#2563EB", edgecolor="#1E40AF", alpha=0.85)
    plt.axvline(median_length, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Median: {median_length:.0f}")
    plt.axvline(mean_length, color="#16A34A", linestyle="-", linewidth=1.5, label=f"Mean: {mean_length:.1f}")
    plt.title("Corpus Document Length Distribution (Word Count)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Words per Document", fontsize=11)
    plt.ylabel("Number of Documents", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(figure_output_path)
    plt.close()
    logger.info("Saved length distribution histogram to %s", figure_output_path)

    # Compile result dictionary
    stats_result: Dict[str, Any] = {
        "document_count": doc_count,
        "total_word_count": total_words,
        "length_distribution": {
            "min": min_length,
            "max": max_length,
            "mean": round(mean_length, 2),
            "median": median_length,
            "std": round(std_length, 2),
            "q25": q25,
            "q75": q75,
        },
        "vocabulary_size": vocab_size,
        "top_terms": [{"term": t, "count": c} for t, c in top_terms],
        "duplicate_rate": round(duplicate_rate, 4),
        "crawler_stats": crawler_stats,
        "internal_duplicates_in_corpus": internal_duplicates,
    }

    # Persist JSON report
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as rf:
        json.dump(stats_result, rf, indent=2)
    logger.info("Saved corpus stats JSON to %s", results_json_path)

    # Print summary table
    print("\n" + "=" * 60)
    print(" " * 18 + "CORPUS STATISTICS REPORT")
    print("=" * 60)
    print(f"Total Documents:       {doc_count:,}")
    print(f"Total Words:           {total_words:,}")
    print(f"Vocabulary Size:       {vocab_size:,} unique tokens")
    print("-" * 60)
    print("Document Length Distribution (words):")
    print(f"  Min:                 {min_length}")
    print(f"  25th Percentile:     {q25}")
    print(f"  Median:              {median_length}")
    print(f"  75th Percentile:     {q75}")
    print(f"  Mean:                {mean_length:.2f} ± {std_length:.2f}")
    print(f"  Max:                 {max_length:,}")
    print("-" * 60)
    print(f"Duplicate Rate:        {duplicate_rate * 100:.2f}%")
    if crawler_stats:
        print(f"  - URL Duplicates:    {crawler_stats.get('duplicates_url', 0)}")
        print(f"  - Content Duplicates:{crawler_stats.get('duplicates_content', 0)}")
        print(f"  - Short Filtered:    {crawler_stats.get('filtered_short', 0)}")
        print(f"  - Lang Filtered:     {crawler_stats.get('filtered_lang', 0)}")
    print("-" * 60)
    print("Top 15 Most Frequent Content Terms:")
    for rank, (term, count) in enumerate(top_terms[:15], 1):
        print(f"  {rank:>2}. {term:<20} {count:>6,}")
    print("=" * 60 + "\n")

    return stats_result


def main() -> None:
    """CLI entrypoint for computing corpus statistics."""
    parser = argparse.ArgumentParser(description="Compute corpus statistics and length distribution")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/documents.jsonl",
        help="Path to documents.jsonl",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="data/raw/crawler_state.json",
        help="Path to crawler_state.json",
    )
    parser.add_argument(
        "--fig-out",
        type=str,
        default="reports/figures/corpus_length_hist.png",
        help="Path to output histogram figure",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="reports/results/corpus_stats.json",
        help="Path to output results JSON",
    )
    args = parser.parse_args()

    compute_statistics(
        jsonl_path=Path(args.input),
        state_path=Path(args.state),
        figure_output_path=Path(args.fig_out),
        results_json_path=Path(args.json_out),
    )


if __name__ == "__main__":
    main()
