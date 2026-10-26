"""Rank and score fusion algorithms for hybrid retrieval.

Includes:
1. Reciprocal Rank Fusion (RRF): Cormack, Clarke, and Buettcher (SIGIR 2009).
2. Min-Max Normalized Weighted Score Fusion: Linear score combination with per-retriever normalization.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Sequence, Tuple


def min_max_normalize(
    scored_items: List[Tuple[str, float]],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Normalize similarity/relevance scores into the [0.0, 1.0] interval.

    Formula:
        norm_score = (score - min_score) / (max_score - min_score + eps)

    If all scores in scored_items are identical or the list contains 1 item,
    all items are assigned 1.0 (or 0.5 if score is 0).

    Args:
        scored_items: List of (doc_id, raw_score) pairs.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dictionary mapping doc_id -> normalized score in [0.0, 1.0].
    """
    if not scored_items:
        return {}

    scores = [s for _, s in scored_items]
    min_s = min(scores)
    max_s = max(scores)
    rng = max_s - min_s

    if rng < eps:
        # Uniform distribution
        return {doc_id: 1.0 for doc_id, _ in scored_items}

    return {doc_id: float((s - min_s) / (rng + eps)) for doc_id, s in scored_items}


def reciprocal_rank_fusion(
    ranked_lists: Sequence[List[Tuple[str, float]]],
    k_rrf: int = 60,
    weights: Optional[Sequence[float]] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Combine multiple ranked candidate lists via Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum_{m in models} w_m * (1 / (k_rrf + rank_m(d)))

    where rank_m(d) is 1-indexed (1 for top result, 2 for second, etc.).

    Args:
        ranked_lists: Sequence of ranked lists, where each list contains
                      (doc_id, score) pairs in descending order of relevance.
        k_rrf: Smoothing constant (default: 60, standard in IR literature).
        weights: Optional list of relative retriever weights w_m. Defaults to uniform (1.0).
        top_k: Optional maximum number of fused results to return.

    Returns:
        Fused list of (doc_id, rrf_score) sorted descending by RRF score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError(f"Length of weights ({len(weights)}) != length of ranked_lists ({len(ranked_lists)})")

    rrf_scores: Dict[str, float] = collections.defaultdict(float)

    for list_idx, r_list in enumerate(ranked_lists):
        w = float(weights[list_idx])
        for rank_zero, (doc_id, _) in enumerate(r_list):
            rank = rank_zero + 1  # 1-indexed rank
            rrf_scores[doc_id] += w * (1.0 / (k_rrf + rank))

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return fused


def score_fusion(
    scored_lists: Sequence[List[Tuple[str, float]]],
    weights: Sequence[float],
    normalize: bool = True,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Combine multiple ranked candidate lists via Weighted Score Fusion.

    Formula:
        Score(d) = sum_{m in models} w_m * norm_score_m(d)

    If a document appears in one retriever's candidate list but not another,
    its missing score in that retriever is imputed as 0.0 (the minimum normalized score).

    Args:
        scored_lists: Sequence of scored lists, where each list contains
                      (doc_id, score) pairs.
        weights: List of weights w_m for each retriever (e.g. [alpha, 1 - alpha]).
        normalize: Whether to apply min-max normalization to each list's scores first.
        top_k: Optional maximum number of fused results to return.

    Returns:
        Fused list of (doc_id, fused_score) sorted descending by score.
    """
    if not scored_lists:
        return []

    if len(weights) != len(scored_lists):
        raise ValueError(f"Length of weights ({len(weights)}) != length of scored_lists ({len(scored_lists)})")

    # Normalize weights so they sum to 1.0 (if sum > 0)
    total_w = sum(weights)
    if total_w > 0:
        norm_weights = [w / total_w for w in weights]
    else:
        norm_weights = [1.0 / len(weights)] * len(weights)

    # Normalize scores per list if requested
    norm_score_maps: List[Dict[str, float]] = []
    for s_list in scored_lists:
        if normalize:
            norm_score_maps.append(min_max_normalize(s_list))
        else:
            norm_score_maps.append({doc_id: float(s) for doc_id, s in s_list})

    # Collect all unique candidate doc_ids
    all_docs = set()
    for s_map in norm_score_maps:
        all_docs.update(s_map.keys())

    fused_scores: Dict[str, float] = collections.defaultdict(float)
    for doc_id in all_docs:
        total_score = 0.0
        for list_idx, s_map in enumerate(norm_score_maps):
            w = norm_weights[list_idx]
            s = s_map.get(doc_id, 0.0)
            total_score += w * s
        fused_scores[doc_id] = total_score

    fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return fused
