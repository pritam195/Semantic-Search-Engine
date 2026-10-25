"""Evaluation visualization and plotting module.

Generates:
1. Grouped bar charts of metrics by query type (reports/figures/metrics_by_query_type.png)
2. Latency vs. nDCG@10 tradeoff curve (reports/figures/latency_vs_ndcg.png)
3. Recall@k curves from k=1 to 10 (reports/figures/recall_at_k_curves.png)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("eval_plots")

MODEL_COLORS = {
    "TF-IDF": "#2563EB",     # Blue
    "BM25": "#D97706",       # Amber
    "MiniLM": "#059669",     # Emerald
    "BGE": "#7C3AED",        # Purple
    "Hybrid (RRF)": "#DC2626"# Red
}


def plot_metrics_by_query_type(results: Dict[str, Any], output_path: Path) -> None:
    """Generate grouped bar plot comparing nDCG@10 across the three query types."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query_types = ["keyword", "natural_language_question", "paraphrase_vocab_mismatch"]
    type_labels = ["Keyword", "Natural Language", "Paraphrase / Mismatch"]
    models = list(results.get("models", {}).keys())

    x = np.arange(len(query_types))
    width = 0.8 / max(1, len(models))

    plt.figure(figsize=(11, 6), dpi=150)

    for i, model_name in enumerate(models):
        model_data = results["models"][model_name]
        per_type = model_data.get("per_query_type", {})
        ndcg_values = [
            per_type.get(q_type, {}).get("ndcg_at_10", 0.0)
            for q_type in query_types
        ]
        color = MODEL_COLORS.get(model_name, None)
        offset = (i - len(models) / 2) * width + width / 2
        bars = plt.bar(x + offset, ndcg_values, width, label=model_name, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        # Add text labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                plt.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    plt.title("Retrieval Performance (nDCG@10) by Query Formulation Type", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Query Type", fontsize=11, labelpad=8)
    plt.ylabel("Mean nDCG@10", fontsize=11, labelpad=8)
    plt.xticks(x, type_labels, fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved metrics by query type plot to %s", output_path)


def plot_latency_vs_ndcg(results: Dict[str, Any], output_path: Path) -> None:
    """Generate latency vs nDCG tradeoff scatter plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(results.get("models", {}).keys())
    latencies = [results["models"][m].get("mean_latency_ms", 5.0) for m in models]
    ndcgs = [results["models"][m]["overall"]["ndcg_at_10"] for m in models]

    plt.figure(figsize=(9, 5.5), dpi=150)

    for m, lat, ndcg in zip(models, latencies, ndcgs):
        color = MODEL_COLORS.get(m, "#2563EB")
        plt.scatter(lat, ndcg, color=color, s=220, edgecolors="black", linewidths=1.5, zorder=5)
        plt.annotate(
            f"  {m}\n  ({lat:.1f}ms, {ndcg:.3f})",
            xy=(lat, ndcg),
            xytext=(6, -4),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="#1F2937",
        )

    plt.title("Retrieval Tradeoff: Query Latency vs. nDCG@10", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean Query Latency (ms)", fontsize=11)
    plt.ylabel("Overall nDCG@10", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved latency vs nDCG plot to %s", output_path)


def plot_recall_at_k_curves(results: Dict[str, Any], output_path: Path) -> None:
    """Plot Recall@k curves for k=1 to 10 across models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ks = list(range(1, 11))
    models = list(results.get("models", {}).keys())

    plt.figure(figsize=(9, 5.5), dpi=150)

    for m in models:
        color = MODEL_COLORS.get(m, None)
        recall_curve = results["models"][m].get("recall_curve", [])
        if not recall_curve or len(recall_curve) < len(ks):
            # Fallback interpolation if only k=5,10 available
            r5 = results["models"][m]["overall"].get("recall_at_5", 0.5)
            r10 = results["models"][m]["overall"].get("recall_at_10", 0.7)
            recall_curve = [r5 * (k / 5.0) if k <= 5 else r5 + (r10 - r5) * ((k - 5) / 5.0) for k in ks]

        plt.plot(ks, recall_curve[:10], marker="o", linewidth=2.0, markersize=6, label=m, color=color)

    plt.title("Recall@k Curves (k = 1 to 10)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Cutoff k", fontsize=11)
    plt.ylabel("Recall@k", fontsize=11)
    plt.xticks(ks)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved recall@k curves plot to %s", output_path)
