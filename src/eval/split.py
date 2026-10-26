"""Stratified Query Splitting for Train/Dev and Held-Out Test Evaluation.

Guarantees:
- Zero data leakage between parameter tuning and test evaluation
- Stratified representation across all query categories (keyword, question, paraphrase)
- Deterministic, reproducible partition using fixed random seed (seed=42)
- Persistence to data/eval/query_splits.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("eval_split")

DEFAULT_QUERIES_PATH = Path("data/eval/queries.jsonl")
DEFAULT_SPLITS_PATH = Path("data/eval/query_splits.json")


def load_queries(queries_path: Path = DEFAULT_QUERIES_PATH) -> List[Dict[str, Any]]:
    """Load query dictionaries from JSONL."""
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def create_stratified_split(
    queries: List[Dict[str, Any]],
    dev_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition queries into stratified dev and test sets.

    Args:
        queries: List of query dicts with 'query_id' and 'query_type'.
        dev_ratio: Fraction of queries allocated to dev split (default 0.5).
        seed: Random seed for reproducibility.

    Returns:
        (dev_queries, test_queries)
    """
    rng = np.random.RandomState(seed)

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for q in queries:
        by_type.setdefault(q["query_type"], []).append(q)

    dev_queries: List[Dict[str, Any]] = []
    test_queries: List[Dict[str, Any]] = []

    for q_type, q_list in sorted(by_type.items()):
        # Sort queries by query_id for deterministic ordering before permutation
        sorted_list = sorted(q_list, key=lambda x: x["query_id"])
        indices = rng.permutation(len(sorted_list))

        n_dev = int(round(len(sorted_list) * dev_ratio))
        dev_idx = set(indices[:n_dev])

        for idx, q in enumerate(sorted_list):
            if idx in dev_idx:
                dev_queries.append(q)
            else:
                test_queries.append(q)

    # Sort each set by query_id
    dev_queries.sort(key=lambda x: x["query_id"])
    test_queries.sort(key=lambda x: x["query_id"])

    logger.info(
        "Created stratified split: Dev=%d queries, Test=%d queries (seed=%d)",
        len(dev_queries),
        len(test_queries),
        seed,
    )
    return dev_queries, test_queries


def save_splits(
    dev_queries: List[Dict[str, Any]],
    test_queries: List[Dict[str, Any]],
    splits_path: Path = DEFAULT_SPLITS_PATH,
) -> None:
    """Save the partition to disk."""
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "num_dev": len(dev_queries),
            "num_test": len(test_queries),
            "dev_ratio": len(dev_queries) / (len(dev_queries) + len(test_queries)),
            "seed": 42,
        },
        "dev_query_ids": [q["query_id"] for q in dev_queries],
        "test_query_ids": [q["query_id"] for q in test_queries],
        "dev_queries": dev_queries,
        "test_queries": test_queries,
    }
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved query splits to %s", splits_path)


def load_or_create_splits(
    queries_path: Path = DEFAULT_QUERIES_PATH,
    splits_path: Path = DEFAULT_SPLITS_PATH,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load existing split from disk or create and persist a new stratified split."""
    if splits_path.exists():
        with open(splits_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["dev_queries"], data["test_queries"]

    queries = load_queries(queries_path)
    dev_q, test_q = create_stratified_split(queries, dev_ratio=0.5, seed=seed)
    save_splits(dev_q, test_q, splits_path)
    return dev_q, test_q
