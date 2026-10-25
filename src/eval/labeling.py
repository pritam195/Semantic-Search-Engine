"""Pooling and relevance judgment generation with CLI and Streamlit labeling tools.

Features:
- Pools top-10 retrieved documents from TF-IDF, BM25, MiniLM, and BGE
- Deduplicates pool per query
- Generates graded relevance judgments: 0 (not relevant), 1 (relevant), 2 (highly relevant)
- Clear separation between unverified suggested judgments and confirmed human labels
- Inter-annotator / Heuristic-vs-Human agreement computation (Cohen's Kappa and % agreement)
- Interactive CLI tool (--cli) and Streamlit Web UI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("eval_labeling")


def compute_agreement(
    judgments_a: Dict[str, Dict[str, int]],
    judgments_b: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """Compute observed agreement and Cohen's Kappa between two sets of annotations.

    Format: {query_id: {doc_id: grade}}
    """
    pairs: List[Tuple[int, int]] = []
    for q_id, docs_a in judgments_a.items():
        if q_id in judgments_b:
            docs_b = judgments_b[q_id]
            for doc_id, grade_a in docs_a.items():
                if doc_id in docs_b:
                    pairs.append((grade_a, docs_b[doc_id]))

    if not pairs:
        return {"observed_agreement": 1.0, "cohen_kappa": 1.0, "total_pairs": 0}

    total = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    observed_agreement = agree / float(total)

    # Cohen's Kappa calculation
    # Marginals for categories 0, 1, 2
    cats = [0, 1, 2]
    pe = 0.0
    for c in cats:
        p_a = sum(1 for a, _ in pairs if a == c) / float(total)
        p_b = sum(1 for _, b in pairs if b == c) / float(total)
        pe += p_a * p_b

    kappa = (observed_agreement - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 1.0

    return {
        "observed_agreement": round(observed_agreement, 4),
        "cohen_kappa": round(kappa, 4),
        "total_pairs": total,
    }


def generate_pooled_judgments(
    queries: List[Dict[str, Any]],
    retrievers_results: Dict[str, Dict[str, List[Tuple[str, float]]]],
    doc_lookup: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    """Build the judgment pool and determine ground truth graded relevance."""
    pool_records: List[Dict[str, Any]] = []

    for q in queries:
        q_id = q["query_id"]
        query_text = q["query"].lower()
        target_topic = q.get("target_topic", "").lower()

        # Pool top-10 from each available retriever
        pooled_doc_ids: Set[str] = set()
        for retriever_name, q_map in retrievers_results.items():
            top_docs = q_map.get(q_id, [])[:10]
            for doc_id, _ in top_docs:
                pooled_doc_ids.add(doc_id)

        for doc_id in pooled_doc_ids:
            doc = doc_lookup.get(doc_id, {})
            title = doc.get("title", "").lower()
            body = (doc.get("clean_body_text") or doc.get("raw_text", "")).lower()

            # Determine graded relevance
            # Grade 2: Highly Relevant (exact title match or primary canonical article)
            # Grade 1: Relevant (topically related or discusses target concepts)
            # Grade 0: Not Relevant
            grade = 0
            if target_topic and (target_topic == title or title == target_topic.split(" (")[0]):
                grade = 2
            elif target_topic and target_topic in title:
                grade = 2
            elif target_topic and target_topic in body[:600]:
                grade = 1
            elif any(w in title for w in query_text.split() if len(w) > 3):
                grade = 1

            pool_records.append({
                "query_id": q_id,
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "relevance": grade,
                "verified": False,
                "annotator": "heuristic_pseudo_judge",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in pool_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Saved %d pooled judgments across %d queries to %s", len(pool_records), len(queries), output_path)
    return pool_records


def run_cli_labeling(qrels_path: Path) -> None:
    """Simple CLI interface for confirming or adjusting judgments."""
    if not qrels_path.exists():
        print(f"Error: {qrels_path} not found.")
        return

    records: List[Dict[str, Any]] = []
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    unverified = [r for r in records if not r.get("verified", False)]
    print(f"\nLoaded {len(records)} judgments ({len(unverified)} unverified).")
    print("Press Enter to accept suggestion, or enter 0, 1, 2, or 'q' to quit.\n")

    for i, r in enumerate(unverified[:15], 1):
        print(f"[{i}/{len(unverified)}] Query ID: {r['query_id']} | Title: {r['title']}")
        print(f"Suggested Grade: {r['relevance']} (0=Not Rel, 1=Relevant, 2=Highly Relevant)")
        choice = input("Your grade [0/1/2/Enter to accept]: ").strip().lower()
        if choice == "q":
            break
        if choice in ("0", "1", "2"):
            r["relevance"] = int(choice)
        r["verified"] = True
        r["annotator"] = "human_user"

    with open(qrels_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved updated judgments to {qrels_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Relevance judgment pooling and labeling tool")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI labeling")
    parser.add_argument("--qrels", type=str, default="data/eval/qrels.jsonl")
    args = parser.parse_args()

    if args.cli:
        run_cli_labeling(Path(args.qrels))
    else:
        print("Use --cli to start CLI labeling or run as part of evaluator.")


if __name__ == "__main__":
    main()
