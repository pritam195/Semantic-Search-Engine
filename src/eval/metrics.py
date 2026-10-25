"""Information retrieval metrics implemented from scratch.

Calculates:
- Precision@k
- Recall@k
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (nDCG@k)
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set


def precision_at_k(retrieved_doc_ids: Sequence[str], relevant_set: Set[str], k: int = 10) -> float:
    """Compute Precision@k: fraction of top-k retrieved documents that are relevant.

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        relevant_set: Set of ground-truth relevant document IDs.
        k: Ranking cutoff.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_doc_ids:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    num_relevant = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return num_relevant / float(k)


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_set: Set[str], k: int = 10) -> float:
    """Compute Recall@k: fraction of all relevant documents captured in top-k.

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        relevant_set: Set of ground-truth relevant document IDs.
        k: Ranking cutoff.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if not relevant_set or k <= 0:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    num_relevant = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return num_relevant / float(len(relevant_set))


def reciprocal_rank(retrieved_doc_ids: Sequence[str], relevant_set: Set[str]) -> float:
    """Compute Reciprocal Rank: inverse rank of the first relevant document retrieved.

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        relevant_set: Set of ground-truth relevant document IDs.

    Returns:
        Reciprocal rank score in [0.0, 1.0].
    """
    if not relevant_set:
        return 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / float(rank)
    return 0.0


def average_precision(retrieved_doc_ids: Sequence[str], relevant_set: Set[str]) -> float:
    """Compute Average Precision (AP) for a single query.

    AP = (1 / |R|) * sum_{i=1}^N (P@i * I(doc_i is relevant))

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        relevant_set: Set of ground-truth relevant document IDs.

    Returns:
        Average precision score in [0.0, 1.0].
    """
    if not relevant_set or not retrieved_doc_ids:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += hits / float(i)

    return sum_precisions / float(len(relevant_set))


def dcg_at_k(retrieved_doc_ids: Sequence[str], graded_relevance: Dict[str, int], k: int = 10) -> float:
    """Compute Discounted Cumulative Gain (DCG@k) using standard gain formula (2^rel - 1).

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        graded_relevance: Mapping of doc_id to integer grade (0, 1, 2, ...).
        k: Ranking cutoff.

    Returns:
        DCG score.
    """
    dcg = 0.0
    cutoff = min(k, len(retrieved_doc_ids))
    for i in range(cutoff):
        doc_id = retrieved_doc_ids[i]
        rel = graded_relevance.get(doc_id, 0)
        if rel > 0:
            gain = (2.0**rel) - 1.0
            discount = math.log2(i + 2.0)  # log2(rank + 1) where rank is 1-indexed -> log2(i + 2)
            dcg += gain / discount
    return dcg


def ndcg_at_k(retrieved_doc_ids: Sequence[str], graded_relevance: Dict[str, int], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@k).

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        graded_relevance: Mapping of doc_id to integer grade.
        k: Ranking cutoff.

    Returns:
        nDCG score in [0.0, 1.0].
    """
    if k <= 0 or not graded_relevance:
        return 0.0

    actual_dcg = dcg_at_k(retrieved_doc_ids, graded_relevance, k=k)

    # Ideal ranking: sort all positive relevance grades descending
    sorted_grades = sorted([g for g in graded_relevance.values() if g > 0], reverse=True)
    if not sorted_grades:
        return 0.0

    ideal_dcg = 0.0
    ideal_cutoff = min(k, len(sorted_grades))
    for i in range(ideal_cutoff):
        gain = (2.0 ** sorted_grades[i]) - 1.0
        discount = math.log2(i + 2.0)
        ideal_dcg += gain / discount

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg
