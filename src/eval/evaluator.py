"""Master evaluation engine for benchmarking lexical, semantic, and hybrid retrievers.

Performs:
- Multi-model retrieval across 90 benchmark queries (30 keyword, 30 question, 30 paraphrase)
- Candidate pooling and graded relevance judgment evaluation
- From-scratch calculation of P@5, P@10, R@5, R@10, MRR, MAP, nDCG@10
- Stratified breakdown by query type
- Paired bootstrap hypothesis tests with p-values and 95% confidence intervals
- Automatic win/loss error analysis extraction
- Generation of comparison plots
- Programmatic JSON results persistence to reports/results/evaluation_results.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from src.eval.labeling import generate_pooled_judgments
from src.eval.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.eval.plots import (
    plot_latency_vs_ndcg,
    plot_metrics_by_query_type,
    plot_recall_at_k_curves,
)
from src.eval.stats import compute_confidence_interval, paired_bootstrap_test
from src.lexical.bm25_retriever import BM25Retriever
from src.lexical.tfidf_retriever import CustomTFIDFRetriever
from src.semantic.embedder import SentenceEmbedder
from src.semantic.faiss_index import FaissIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluator")


def evaluate_ranking(
    retrieved_doc_ids: List[str],
    graded_relevance: Dict[str, int],
    relevant_set: Set[str],
) -> Dict[str, Any]:
    """Compute all evaluation metrics for a single query."""
    p5 = precision_at_k(retrieved_doc_ids, relevant_set, k=5)
    p10 = precision_at_k(retrieved_doc_ids, relevant_set, k=10)
    r5 = recall_at_k(retrieved_doc_ids, relevant_set, k=5)
    r10 = recall_at_k(retrieved_doc_ids, relevant_set, k=10)
    rr = reciprocal_rank(retrieved_doc_ids, relevant_set)
    ap = average_precision(retrieved_doc_ids, relevant_set)
    ndcg10 = ndcg_at_k(retrieved_doc_ids, graded_relevance, k=10)

    # Recall curve from k=1 to 10
    recall_curve = [recall_at_k(retrieved_doc_ids, relevant_set, k=i) for i in range(1, 11)]

    return {
        "p@5": p5,
        "p@10": p10,
        "r@5": r5,
        "r@10": r10,
        "mrr": rr,
        "map": ap,
        "ndcg@10": ndcg10,
        "recall_curve": recall_curve,
    }


def run_evaluation(
    config_path: str = "configs/eval_config.yaml",
    max_corpus_docs: int = 5000,
) -> Dict[str, Any]:
    """Run full benchmark evaluation across TF-IDF, BM25, MiniLM, and BGE."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    queries_path = Path(config.get("queries_path", "data/eval/queries.jsonl"))
    qrels_path = Path(config.get("qrels_path", "data/eval/qrels.jsonl"))
    results_output = Path(config.get("results_output", "reports/results/evaluation_results.json"))
    figures_dir = Path(config.get("figures_dir", "reports/figures"))
    cache_dir = Path("data/indexes/embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Queries
    logger.info("Loading queries from %s...", queries_path)
    queries: List[Dict[str, Any]] = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    logger.info("Loaded %d queries across 3 categories.", len(queries))

    # 2. Load Corpus
    corpus_file = Path("data/raw/documents.jsonl")
    logger.info("Loading documents from %s (up to %d)...", corpus_file, max_corpus_docs)
    corpus: List[Dict[str, Any]] = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
            if len(corpus) >= max_corpus_docs:
                break
    doc_lookup = {d["doc_id"]: d for d in corpus}
    doc_ids = [d["doc_id"] for d in corpus]
    logger.info("Corpus loaded: %d documents.", len(corpus))

    # Prepare document texts: title + clean text summary
    doc_texts = [d["title"] + ". " + d["clean_body_text"][:600] for d in corpus]

    # 3. Initialize Retrievers
    logger.info("Initializing retrievers on the shared 5,000 document collection...")
    retrievers: Dict[str, Any] = {}

    # A. Custom TF-IDF
    logger.info("Indexing Custom TF-IDF on corpus...")
    tfidf = CustomTFIDFRetriever(sublinear_tf=True, smooth_idf=True, min_df=2)
    tfidf.index(corpus)
    retrievers["TF-IDF"] = tfidf

    # B. BM25
    logger.info("Indexing BM25 on corpus...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(corpus)
    retrievers["BM25"] = bm25

    # C. MiniLM
    logger.info("Setting up MiniLM dense retriever...")
    minilm_embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    minilm_cache = cache_dir / f"minilm_doc_embeddings_{len(corpus)}.npy"
    if minilm_cache.exists():
        logger.info("Loading cached MiniLM embeddings from %s...", minilm_cache)
        minilm_embs = SentenceEmbedder.load_cache(minilm_cache)
    else:
        logger.info("Computing MiniLM embeddings for %d documents...", len(corpus))
        minilm_embs = minilm_embedder.embed_passages(doc_texts, batch_size=128)
        SentenceEmbedder.save_cache(minilm_embs, minilm_cache)

    minilm_faiss = FaissIndex(dimension=minilm_embs.shape[1], index_type="exact")
    minilm_faiss.add(minilm_embs)

    class DenseSearcher:
        def __init__(self, embedder: SentenceEmbedder, index: FaissIndex, id_list: List[str]):
            self.embedder = embedder
            self.index = index
            self.id_list = id_list

        def search(self, q_text: str, k: int = 10) -> List[Tuple[str, float]]:
            q_vec = self.embedder.embed_query(q_text)
            scores, indices = self.index.search(q_vec, k=k)
            res = []
            for s, idx in zip(scores, indices):
                if 0 <= idx < len(self.id_list):
                    res.append((self.id_list[idx], float(s)))
            return res

    retrievers["MiniLM"] = DenseSearcher(minilm_embedder, minilm_faiss, doc_ids)

    # D. BGE
    logger.info("Setting up BGE dense retriever...")
    bge_embedder = SentenceEmbedder(model_name="BAAI/bge-small-en-v1.5")
    bge_cache = cache_dir / f"bge_doc_embeddings_{len(corpus)}.npy"
    if bge_cache.exists():
        logger.info("Loading cached BGE embeddings from %s...", bge_cache)
        bge_embs = SentenceEmbedder.load_cache(bge_cache)
    else:
        logger.info("Computing BGE embeddings for %d documents...", len(corpus))
        bge_embs = bge_embedder.embed_passages(doc_texts, batch_size=128)
        SentenceEmbedder.save_cache(bge_embs, bge_cache)

    bge_faiss = FaissIndex(dimension=bge_embs.shape[1], index_type="exact")
    bge_faiss.add(bge_embs)
    retrievers["BGE"] = DenseSearcher(bge_embedder, bge_faiss, doc_ids)

    # 4. Generate Predictions from each Retriever
    logger.info("Running retrievers on all %d queries...", len(queries))
    predictions: Dict[str, Dict[str, List[Tuple[str, float]]]] = collections.defaultdict(dict)
    latencies: Dict[str, List[float]] = collections.defaultdict(list)

    for model_name, ret in retrievers.items():
        logger.info("Executing retrieval for %s...", model_name)
        for q in queries:
            q_id = q["query_id"]
            t0 = time.perf_counter()
            top_docs = ret.search(q["query"], k=10)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            predictions[model_name][q_id] = top_docs
            latencies[model_name].append(lat_ms)

    # 5. Build or Load Pooled Relevance Judgments
    if not qrels_path.exists():
        logger.info("Generating pooled judgments to %s...", qrels_path)
        generate_pooled_judgments(queries, predictions, doc_lookup, qrels_path)

    # Load judgments: {query_id: {doc_id: grade}}
    qrels: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qrels[item["query_id"]][item["doc_id"]] = item["relevance"]

    # 6. Compute Metrics Per Model, Overall and Per Query Type
    logger.info("Computing metrics across models and query categories...")
    model_eval: Dict[str, Any] = {}
    query_type_map = {q["query_id"]: q["query_type"] for q in queries}
    query_types = ["keyword", "natural_language_question", "paraphrase_vocab_mismatch"]

    # Store query-level scores for statistical tests: {model: [ndcg_q1, ndcg_q2, ...]}
    model_query_ndcg: Dict[str, List[float]] = {}
    model_query_map: Dict[str, List[float]] = {}

    for model_name in retrievers.keys():
        overall_metrics: Dict[str, List[float]] = collections.defaultdict(list)
        type_metrics: Dict[str, Dict[str, List[float]]] = {
            t: collections.defaultdict(list) for t in query_types
        }

        for q in queries:
            q_id = q["query_id"]
            q_type = query_type_map[q_id]
            retrieved_ids = [doc_id for doc_id, _ in predictions[model_name][q_id]]

            graded = qrels.get(q_id, {})
            relevant_set = set(doc_id for doc_id, grade in graded.items() if grade >= 1)

            res = evaluate_ranking(retrieved_ids, graded, relevant_set)

            for m_key, val in res.items():
                if m_key != "recall_curve":
                    overall_metrics[m_key].append(val)
                    type_metrics[q_type][m_key].append(val)
                else:
                    overall_metrics[m_key].append(val)

        # Aggregate overall
        mean_ndcg = float(np.mean(overall_metrics["ndcg@10"]))
        ci_lower, ci_upper = compute_confidence_interval(overall_metrics["ndcg@10"])
        model_query_ndcg[model_name] = overall_metrics["ndcg@10"]
        model_query_map[model_name] = overall_metrics["map"]

        avg_recall_curve = list(np.mean(np.array(overall_metrics["recall_curve"]), axis=0))

        model_summary = {
            "mean_latency_ms": round(float(np.mean(latencies[model_name])), 2),
            "overall": {
                "precision_at_5": round(float(np.mean(overall_metrics["p@5"])), 4),
                "precision_at_10": round(float(np.mean(overall_metrics["p@10"])), 4),
                "recall_at_5": round(float(np.mean(overall_metrics["r@5"])), 4),
                "recall_at_10": round(float(np.mean(overall_metrics["r@10"])), 4),
                "mrr": round(float(np.mean(overall_metrics["mrr"])), 4),
                "map": round(float(np.mean(overall_metrics["map"])), 4),
                "ndcg_at_10": round(mean_ndcg, 4),
                "ndcg_95_ci": [round(ci_lower, 4), round(ci_upper, 4)],
            },
            "recall_curve": [round(float(x), 4) for x in avg_recall_curve],
            "per_query_type": {},
        }

        for q_type in query_types:
            t_data = type_metrics[q_type]
            model_summary["per_query_type"][q_type] = {
                "precision_at_10": round(float(np.mean(t_data["p@10"])), 4),
                "recall_at_10": round(float(np.mean(t_data["r@10"])), 4),
                "mrr": round(float(np.mean(t_data["mrr"])), 4),
                "map": round(float(np.mean(t_data["map"])), 4),
                "ndcg_at_10": round(float(np.mean(t_data["ndcg@10"])), 4),
            }

        model_eval[model_name] = model_summary

    # 7. Statistical Significance Tests
    logger.info("Conducting paired bootstrap significance tests...")
    statistical_tests = {
        "BM25_vs_TFIDF": paired_bootstrap_test(model_query_ndcg["BM25"], model_query_ndcg["TF-IDF"]),
        "MiniLM_vs_BM25": paired_bootstrap_test(model_query_ndcg["MiniLM"], model_query_ndcg["BM25"]),
        "BGE_vs_BM25": paired_bootstrap_test(model_query_ndcg["BGE"], model_query_ndcg["BM25"]),
        "BGE_vs_MiniLM": paired_bootstrap_test(model_query_ndcg["BGE"], model_query_ndcg["MiniLM"]),
    }

    # 8. Error Analysis Extraction
    logger.info("Extracting query win/loss error cases...")
    diff_records = []
    for i, q in enumerate(queries):
        bm25_score = model_query_ndcg["BM25"][i]
        minilm_score = model_query_ndcg["MiniLM"][i]
        diff_records.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "query_type": q["query_type"],
            "target_topic": q.get("target_topic", ""),
            "bm25_ndcg": round(bm25_score, 4),
            "minilm_ndcg": round(minilm_score, 4),
            "diff_bm25_minus_minilm": round(bm25_score - minilm_score, 4),
        })

    # Sort for Lexical (BM25) wins
    diff_records.sort(key=lambda x: x["diff_bm25_minus_minilm"], reverse=True)
    lexical_wins = diff_records[:10]

    # Sort for Semantic (MiniLM) wins
    diff_records.sort(key=lambda x: x["diff_bm25_minus_minilm"])
    semantic_wins = diff_records[:10]

    # 9. Compile Complete Report
    final_report = {
        "benchmark_summary": {
            "num_queries": len(queries),
            "query_types": {t: sum(1 for q in queries if q["query_type"] == t) for t in query_types},
            "corpus_documents": len(corpus),
            "models_evaluated": list(retrievers.keys()),
        },
        "models": model_eval,
        "statistical_significance": statistical_tests,
        "error_analysis": {
            "top_10_lexical_wins": lexical_wins,
            "top_10_semantic_wins": semantic_wins,
        },
    }

    with open(results_output, "w", encoding="utf-8") as rf:
        json.dump(final_report, rf, indent=2)
    logger.info("Saved complete evaluation report to %s", results_output)

    # 10. Generate Figures
    logger.info("Generating evaluation figures in %s...", figures_dir)
    plot_metrics_by_query_type(final_report, figures_dir / "metrics_by_query_type.png")
    plot_latency_vs_ndcg(final_report, figures_dir / "latency_vs_ndcg.png")
    plot_recall_at_k_curves(final_report, figures_dir / "recall_at_k_curves.png")

    # 11. Print Formatted Benchmark Table
    print("\n" + "=" * 85)
    print(" " * 26 + "RIGOROUS RETRIEVAL EVALUATION REPORT")
    print("=" * 85)
    print(f"Total Queries: {len(queries)} (30 Keyword | 30 Question | 30 Paraphrase Mismatch)")
    print("-" * 85)
    print(f"{'Retriever':<15} | {'nDCG@10':<9} | {'95% CI':<16} | {'MAP':<8} | {'MRR':<8} | {'P@10':<8} | {'R@10':<8}")
    print("-" * 85)
    for m in retrievers.keys():
        o = model_eval[m]["overall"]
        ci_str = f"[{o['ndcg_95_ci'][0]:.3f}, {o['ndcg_95_ci'][1]:.3f}]"
        print(f"{m:<15} | {o['ndcg_at_10']:<9.4f} | {ci_str:<16} | {o['map']:<8.4f} | {o['mrr']:<8.4f} | {o['precision_at_10']:<8.4f} | {o['recall_at_10']:<8.4f}")
    print("-" * 85)
    print("PER QUERY-TYPE PERFORMANCE (nDCG@10):")
    print(f"{'Retriever':<15} | {'Keyword':<12} | {'Question':<12} | {'Paraphrase / Mismatch':<22}")
    print("-" * 85)
    for m in retrievers.keys():
        pt = model_eval[m]["per_query_type"]
        print(f"{m:<15} | {pt['keyword']['ndcg_at_10']:<12.4f} | {pt['natural_language_question']['ndcg_at_10']:<12.4f} | {pt['paraphrase_vocab_mismatch']['ndcg_at_10']:<22.4f}")
    print("-" * 85)
    print("STATISTICAL SIGNIFICANCE (Paired Bootstrap, B=1000):")
    for comp, stats_data in statistical_tests.items():
        sig_mark = "*** (p<0.01)" if stats_data["statistically_significant_01"] else ("* (p<0.05)" if stats_data["statistically_significant_05"] else "n.s.")
        print(f"  {comp:<20}: diff={stats_data['mean_diff']:+.4f}, p={stats_data['bootstrap_p_value']:.4f} {sig_mark}")
    print("=" * 85 + "\n")

    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Master IR evaluation suite")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument("--max-docs", type=int, default=5000)
    args = parser.parse_args()

    run_evaluation(config_path=args.config, max_corpus_docs=args.max_docs)


if __name__ == "__main__":
    main()
