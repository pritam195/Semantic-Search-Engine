"""Hybrid Retriever Parameter Tuning & Held-Out Test Evaluation.

Enforces:
1. Strict train/dev and held-out test split (45 dev queries, 45 test queries, stratified).
2. Tuning of score fusion weight (alpha) and RRF smoothing constant (k_rrf) on DEV ONLY.
3. Final evaluation of frozen best parameters on HELD-OUT TEST split (no leakage).
4. Cross-encoder neural reranking (ms-marco-MiniLM-L-6-v2) on top 50 candidates.
5. Paired bootstrap hypothesis testing (B=1000) and 95% confidence intervals.
6. Generation of tuning plots and persistent JSON results.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from src.eval.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.eval.split import load_or_create_splits
from src.eval.stats import compute_confidence_interval, paired_bootstrap_test
from src.hybrid.fusion import reciprocal_rank_fusion, score_fusion
from src.hybrid.reranker import CrossEncoderReranker
from src.hybrid.retriever import HybridRetriever
from src.lexical.bm25_retriever import BM25Retriever
from src.lexical.tfidf_retriever import CustomTFIDFRetriever
from src.semantic.embedder import SentenceEmbedder
from src.semantic.faiss_index import FaissIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("hybrid_tune")


class DenseSearcher:
    """Wrapper for FAISS dense vector search over corpus doc IDs."""

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


def evaluate_query_ranking(
    retrieved_doc_ids: List[str],
    graded_relevance: Dict[str, int],
    relevant_set: Set[str],
) -> Dict[str, float]:
    """Compute standard IR metrics for a single query."""
    return {
        "p@5": precision_at_k(retrieved_doc_ids, relevant_set, k=5),
        "p@10": precision_at_k(retrieved_doc_ids, relevant_set, k=10),
        "r@5": recall_at_k(retrieved_doc_ids, relevant_set, k=5),
        "r@10": recall_at_k(retrieved_doc_ids, relevant_set, k=10),
        "mrr": reciprocal_rank(retrieved_doc_ids, relevant_set),
        "map": average_precision(retrieved_doc_ids, relevant_set),
        "ndcg@10": ndcg_at_k(retrieved_doc_ids, graded_relevance, k=10),
    }


def aggregate_query_metrics(
    query_list: List[Dict[str, Any]],
    predictions: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """Compute aggregate metrics and confidence intervals over a query split."""
    metric_lists: Dict[str, List[float]] = collections.defaultdict(list)
    by_type: Dict[str, Dict[str, List[float]]] = {
        "keyword": collections.defaultdict(list),
        "natural_language_question": collections.defaultdict(list),
        "paraphrase_vocab_mismatch": collections.defaultdict(list),
    }

    for q in query_list:
        q_id = q["query_id"]
        q_type = q["query_type"]
        retrieved_ids = [doc_id for doc_id, _ in predictions.get(q_id, [])]

        graded = qrels.get(q_id, {})
        relevant_set = set(doc_id for doc_id, grade in graded.items() if grade >= 1)

        res = evaluate_query_ranking(retrieved_ids, graded, relevant_set)
        for m, val in res.items():
            metric_lists[m].append(val)
            by_type[q_type][m].append(val)

    mean_ndcg = float(np.mean(metric_lists["ndcg@10"])) if metric_lists["ndcg@10"] else 0.0
    ci_lower, ci_upper = compute_confidence_interval(metric_lists["ndcg@10"]) if metric_lists["ndcg@10"] else (0.0, 0.0)

    res_summary = {
        "precision_at_5": round(float(np.mean(metric_lists["p@5"])), 4),
        "precision_at_10": round(float(np.mean(metric_lists["p@10"])), 4),
        "recall_at_5": round(float(np.mean(metric_lists["r@5"])), 4),
        "recall_at_10": round(float(np.mean(metric_lists["r@10"])), 4),
        "mrr": round(float(np.mean(metric_lists["mrr"])), 4),
        "map": round(float(np.mean(metric_lists["map"])), 4),
        "ndcg_at_10": round(mean_ndcg, 4),
        "ndcg_95_ci": [round(ci_lower, 4), round(ci_upper, 4)],
        "per_query_ndcg": metric_lists["ndcg@10"],
        "per_query_map": metric_lists["map"],
        "by_query_type": {},
    }

    for t_name, t_metrics in by_type.items():
        if t_metrics["ndcg@10"]:
            res_summary["by_query_type"][t_name] = {
                "precision_at_10": round(float(np.mean(t_metrics["p@10"])), 4),
                "recall_at_10": round(float(np.mean(t_metrics["r@10"])), 4),
                "mrr": round(float(np.mean(t_metrics["mrr"])), 4),
                "map": round(float(np.mean(t_metrics["map"])), 4),
                "ndcg_at_10": round(float(np.mean(t_metrics["ndcg@10"])), 4),
            }

    return res_summary


def run_hybrid_tuning(
    config_path: str = "configs/eval_config.yaml",
    max_corpus_docs: int = 5000,
) -> Dict[str, Any]:
    """Execute tuning on dev split and rigorous evaluation on test split."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    queries_path = Path(config.get("queries_path", "data/eval/queries.jsonl"))
    qrels_path = Path(config.get("qrels_path", "data/eval/qrels.jsonl"))
    splits_path = Path("data/eval/query_splits.json")
    results_dir = Path("reports/results")
    figures_dir = Path("reports/figures")
    cache_dir = Path("data/indexes/embeddings")

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Dev and Test query splits
    logger.info("Loading query splits from %s...", splits_path)
    dev_queries, test_queries = load_or_create_splits(queries_path, splits_path, seed=42)
    all_queries = dev_queries + test_queries
    logger.info("Dev queries: %d, Test queries: %d, Total: %d", len(dev_queries), len(test_queries), len(all_queries))

    # 2. Load Corpus
    corpus_file = Path("data/raw/documents.jsonl")
    logger.info("Loading documents from %s (up to %d)...", corpus_file, max_corpus_docs)
    corpus = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
            if len(corpus) >= max_corpus_docs:
                break
    doc_lookup = {d["doc_id"]: d for d in corpus}
    doc_ids = [d["doc_id"] for d in corpus]
    doc_texts = [d["title"] + ". " + d["clean_body_text"][:600] for d in corpus]
    logger.info("Corpus loaded: %d documents.", len(corpus))

    # 3. Load Qrels
    logger.info("Loading ground truth judgments from %s...", qrels_path)
    qrels: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qrels[item["query_id"]][item["doc_id"]] = item["relevance"]

    # 4. Initialize Base Retrievers
    logger.info("Initializing base retrievers on corpus...")
    # Lexical: BM25 and TF-IDF
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(corpus)

    tfidf = CustomTFIDFRetriever(sublinear_tf=True, smooth_idf=True, min_df=2)
    tfidf.index(corpus)

    # Dense: MiniLM
    minilm_embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    minilm_cache = cache_dir / f"minilm_doc_embeddings_{len(corpus)}.npy"
    if minilm_cache.exists():
        minilm_embs = SentenceEmbedder.load_cache(minilm_cache)
    else:
        minilm_embs = minilm_embedder.embed_passages(doc_texts, batch_size=128)
        SentenceEmbedder.save_cache(minilm_embs, minilm_cache)
    minilm_faiss = FaissIndex(dimension=minilm_embs.shape[1], index_type="exact")
    minilm_faiss.add(minilm_embs)
    minilm_searcher = DenseSearcher(minilm_embedder, minilm_faiss, doc_ids)

    # Dense: BGE
    bge_embedder = SentenceEmbedder(model_name="BAAI/bge-small-en-v1.5")
    bge_cache = cache_dir / f"bge_doc_embeddings_{len(corpus)}.npy"
    if bge_cache.exists():
        bge_embs = SentenceEmbedder.load_cache(bge_cache)
    else:
        bge_embs = bge_embedder.embed_passages(doc_texts, batch_size=128)
        SentenceEmbedder.save_cache(bge_embs, bge_cache)
    bge_faiss = FaissIndex(dimension=bge_embs.shape[1], index_type="exact")
    bge_faiss.add(bge_embs)
    bge_searcher = DenseSearcher(bge_embedder, bge_faiss, doc_ids)

    # Cross-Encoder Reranker
    reranker = CrossEncoderReranker(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

    # 5. Pre-retrieve candidate pools (top 100) for all queries to make tuning blazing fast
    logger.info("Pre-retrieving candidate pools (top 100) across all queries...")
    pool_bm25: Dict[str, List[Tuple[str, float]]] = {}
    pool_tfidf: Dict[str, List[Tuple[str, float]]] = {}
    pool_minilm: Dict[str, List[Tuple[str, float]]] = {}
    pool_bge: Dict[str, List[Tuple[str, float]]] = {}

    for q in all_queries:
        qid = q["query_id"]
        qtxt = q["query"]
        pool_bm25[qid] = bm25.search(qtxt, k=100)
        pool_tfidf[qid] = tfidf.search(qtxt, k=100)
        pool_minilm[qid] = minilm_searcher.search(qtxt, k=100)
        pool_bge[qid] = bge_searcher.search(qtxt, k=100)

    # ---------------------------------------------------------
    # STEP 1: HYPERPARAMETER TUNING ON DEV SPLIT (ZERO TEST LEAKAGE)
    # ---------------------------------------------------------
    logger.info("--- STEP 1: TUNING HYPERPARAMETERS ON DEV SPLIT (N=%d) ---", len(dev_queries))

    # A. Sweep alpha for BM25 + MiniLM Score Fusion
    alpha_grid = [round(a, 2) for a in np.linspace(0.0, 1.0, 21)]
    dev_alpha_results_minilm: Dict[float, Dict[str, Any]] = {}
    test_alpha_results_minilm: Dict[float, Dict[str, Any]] = {}  # for plotting generalization curve

    for alpha in alpha_grid:
        dev_preds: Dict[str, List[Tuple[str, float]]] = {}
        for q in dev_queries:
            qid = q["query_id"]
            dev_preds[qid] = score_fusion([pool_bm25[qid], pool_minilm[qid]], weights=[alpha, 1.0 - alpha], top_k=10)
        dev_eval = aggregate_query_metrics(dev_queries, dev_preds, qrels)
        dev_alpha_results_minilm[alpha] = {
            "ndcg@10": dev_eval["ndcg_at_10"],
            "map": dev_eval["map"],
            "mrr": dev_eval["mrr"],
        }

        # Also compute on test for visualization
        test_preds: Dict[str, List[Tuple[str, float]]] = {}
        for q in test_queries:
            qid = q["query_id"]
            test_preds[qid] = score_fusion([pool_bm25[qid], pool_minilm[qid]], weights=[alpha, 1.0 - alpha], top_k=10)
        test_eval = aggregate_query_metrics(test_queries, test_preds, qrels)
        test_alpha_results_minilm[alpha] = {
            "ndcg@10": test_eval["ndcg_at_10"],
            "map": test_eval["map"],
            "mrr": test_eval["mrr"],
        }

    # Find best alpha for BM25 + MiniLM on DEV ONLY
    best_alpha_minilm = max(dev_alpha_results_minilm.keys(), key=lambda a: dev_alpha_results_minilm[a]["ndcg@10"])
    logger.info(
        "Optimal alpha for BM25 + MiniLM on Dev: alpha=%.2f (Dev nDCG@10=%.4f)",
        best_alpha_minilm,
        dev_alpha_results_minilm[best_alpha_minilm]["ndcg@10"],
    )

    # B. Sweep alpha for BM25 + BGE Score Fusion
    dev_alpha_results_bge: Dict[float, Dict[str, Any]] = {}
    test_alpha_results_bge: Dict[float, Dict[str, Any]] = {}

    for alpha in alpha_grid:
        dev_preds = {}
        for q in dev_queries:
            qid = q["query_id"]
            dev_preds[qid] = score_fusion([pool_bm25[qid], pool_bge[qid]], weights=[alpha, 1.0 - alpha], top_k=10)
        dev_eval = aggregate_query_metrics(dev_queries, dev_preds, qrels)
        dev_alpha_results_bge[alpha] = {
            "ndcg@10": dev_eval["ndcg_at_10"],
            "map": dev_eval["map"],
            "mrr": dev_eval["mrr"],
        }

        test_preds = {}
        for q in test_queries:
            qid = q["query_id"]
            test_preds[qid] = score_fusion([pool_bm25[qid], pool_bge[qid]], weights=[alpha, 1.0 - alpha], top_k=10)
        test_eval = aggregate_query_metrics(test_queries, test_preds, qrels)
        test_alpha_results_bge[alpha] = {
            "ndcg@10": test_eval["ndcg_at_10"],
            "map": test_eval["map"],
            "mrr": test_eval["mrr"],
        }

    best_alpha_bge = max(dev_alpha_results_bge.keys(), key=lambda a: dev_alpha_results_bge[a]["ndcg@10"])
    logger.info(
        "Optimal alpha for BM25 + BGE on Dev: alpha=%.2f (Dev nDCG@10=%.4f)",
        best_alpha_bge,
        dev_alpha_results_bge[best_alpha_bge]["ndcg@10"],
    )

    # C. Sweep k_rrf for RRF
    k_rrf_grid = [10, 20, 30, 40, 50, 60, 80, 100, 120]
    dev_rrf_results_minilm: Dict[int, float] = {}
    dev_rrf_results_bge: Dict[int, float] = {}

    for k_val in k_rrf_grid:
        dev_preds_m = {
            q["query_id"]: reciprocal_rank_fusion([pool_bm25[q["query_id"]], pool_minilm[q["query_id"]]], k_rrf=k_val, top_k=10)
            for q in dev_queries
        }
        dev_rrf_results_minilm[k_val] = aggregate_query_metrics(dev_queries, dev_preds_m, qrels)["ndcg_at_10"]

        dev_preds_b = {
            q["query_id"]: reciprocal_rank_fusion([pool_bm25[q["query_id"]], pool_bge[q["query_id"]]], k_rrf=k_val, top_k=10)
            for q in dev_queries
        }
        dev_rrf_results_bge[k_val] = aggregate_query_metrics(dev_queries, dev_preds_b, qrels)["ndcg_at_10"]

    best_k_rrf_minilm = max(dev_rrf_results_minilm.keys(), key=lambda k: dev_rrf_results_minilm[k])
    best_k_rrf_bge = max(dev_rrf_results_bge.keys(), key=lambda k: dev_rrf_results_bge[k])
    logger.info(
        "Optimal k_rrf on Dev: MiniLM k=%d (nDCG=%.4f), BGE k=%d (nDCG=%.4f)",
        best_k_rrf_minilm,
        dev_rrf_results_minilm[best_k_rrf_minilm],
        best_k_rrf_bge,
        dev_rrf_results_bge[best_k_rrf_bge],
    )

    # ---------------------------------------------------------
    # STEP 2: EVALUATION ON HELD-OUT TEST SPLIT (N=45)
    # ---------------------------------------------------------
    logger.info("--- STEP 2: RIGOROUS EVALUATION ON HELD-OUT TEST SPLIT (N=%d) ---", len(test_queries))

    # Define test models
    # Generate predictions on test set:
    test_models_preds: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

    # Baselines
    test_models_preds["TF-IDF"] = {q["query_id"]: pool_tfidf[q["query_id"]][:10] for q in test_queries}
    test_models_preds["BM25"] = {q["query_id"]: pool_bm25[q["query_id"]][:10] for q in test_queries}
    test_models_preds["MiniLM"] = {q["query_id"]: pool_minilm[q["query_id"]][:10] for q in test_queries}
    test_models_preds["BGE"] = {q["query_id"]: pool_bge[q["query_id"]][:10] for q in test_queries}

    # Hybrid RRF (tuned on dev)
    test_models_preds["Hybrid_RRF_MiniLM"] = {
        q["query_id"]: reciprocal_rank_fusion([pool_bm25[q["query_id"]], pool_minilm[q["query_id"]]], k_rrf=best_k_rrf_minilm, top_k=10)
        for q in test_queries
    }
    test_models_preds["Hybrid_RRF_BGE"] = {
        q["query_id"]: reciprocal_rank_fusion([pool_bm25[q["query_id"]], pool_bge[q["query_id"]]], k_rrf=best_k_rrf_bge, top_k=10)
        for q in test_queries
    }

    # Hybrid Score Fusion (tuned alpha on dev)
    test_models_preds["Hybrid_ScoreFusion_MiniLM"] = {
        q["query_id"]: score_fusion([pool_bm25[q["query_id"]], pool_minilm[q["query_id"]]], weights=[best_alpha_minilm, 1.0 - best_alpha_minilm], top_k=10)
        for q in test_queries
    }
    test_models_preds["Hybrid_ScoreFusion_BGE"] = {
        q["query_id"]: score_fusion([pool_bm25[q["query_id"]], pool_bge[q["query_id"]]], weights=[best_alpha_bge, 1.0 - best_alpha_bge], top_k=10)
        for q in test_queries
    }

    # Cross-Encoder Reranking on top 50 candidates from best hybrid
    logger.info("Executing Cross-Encoder reranking on top 50 candidates for Test queries...")
    rerank_preds_test: Dict[str, List[Tuple[str, float]]] = {}
    for q in test_queries:
        qid = q["query_id"]
        # Take top 50 candidates from Hybrid ScoreFusion BGE
        top50 = score_fusion([pool_bm25[qid], pool_bge[qid]], weights=[best_alpha_bge, 1.0 - best_alpha_bge], top_k=50)
        cand_ids = [doc_id for doc_id, _ in top50]
        rerank_preds_test[qid] = reranker.rerank_doc_lookup(q["query"], cand_ids, doc_lookup, top_k=10)
    test_models_preds["Hybrid_BGE_CrossEncoder"] = rerank_preds_test

    # Compute Test Split Metrics
    test_eval_results: Dict[str, Any] = {}
    for model_name, preds in test_models_preds.items():
        summary = aggregate_query_metrics(test_queries, preds, qrels)
        test_eval_results[model_name] = summary

    # Significance Tests on Held-Out Test Set (Paired Bootstrap, B=1000)
    test_significance = {
        "Hybrid_RRF_BGE_vs_BM25": paired_bootstrap_test(
            test_eval_results["Hybrid_RRF_BGE"]["per_query_ndcg"],
            test_eval_results["BM25"]["per_query_ndcg"],
        ),
        "Hybrid_ScoreFusion_BGE_vs_BM25": paired_bootstrap_test(
            test_eval_results["Hybrid_ScoreFusion_BGE"]["per_query_ndcg"],
            test_eval_results["BM25"]["per_query_ndcg"],
        ),
        "Hybrid_ScoreFusion_BGE_vs_BGE": paired_bootstrap_test(
            test_eval_results["Hybrid_ScoreFusion_BGE"]["per_query_ndcg"],
            test_eval_results["BGE"]["per_query_ndcg"],
        ),
        "Hybrid_BGE_CrossEncoder_vs_Hybrid_ScoreFusion_BGE": paired_bootstrap_test(
            test_eval_results["Hybrid_BGE_CrossEncoder"]["per_query_ndcg"],
            test_eval_results["Hybrid_ScoreFusion_BGE"]["per_query_ndcg"],
        ),
        "Hybrid_BGE_CrossEncoder_vs_BM25": paired_bootstrap_test(
            test_eval_results["Hybrid_BGE_CrossEncoder"]["per_query_ndcg"],
            test_eval_results["BM25"]["per_query_ndcg"],
        ),
    }

    # ---------------------------------------------------------
    # STEP 3: EVALUATION OVER ALL 90 QUERIES (COMPREHENSIVE BENCHMARK)
    # ---------------------------------------------------------
    logger.info("--- STEP 3: OVERALL EVALUATION ACROSS ALL 90 QUERIES ---")
    all_models_preds: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    all_models_preds["TF-IDF"] = {q["query_id"]: pool_tfidf[q["query_id"]][:10] for q in all_queries}
    all_models_preds["BM25"] = {q["query_id"]: pool_bm25[q["query_id"]][:10] for q in all_queries}
    all_models_preds["MiniLM"] = {q["query_id"]: pool_minilm[q["query_id"]][:10] for q in all_queries}
    all_models_preds["BGE"] = {q["query_id"]: pool_bge[q["query_id"]][:10] for q in all_queries}
    all_models_preds["Hybrid_RRF_MiniLM"] = {
        q["query_id"]: reciprocal_rank_fusion([pool_bm25[q["query_id"]], pool_minilm[q["query_id"]]], k_rrf=best_k_rrf_minilm, top_k=10)
        for q in all_queries
    }
    all_models_preds["Hybrid_RRF_BGE"] = {
        q["query_id"]: reciprocal_rank_fusion([pool_bm25[q["query_id"]], pool_bge[q["query_id"]]], k_rrf=best_k_rrf_bge, top_k=10)
        for q in all_queries
    }
    all_models_preds["Hybrid_ScoreFusion_MiniLM"] = {
        q["query_id"]: score_fusion([pool_bm25[q["query_id"]], pool_minilm[q["query_id"]]], weights=[best_alpha_minilm, 1.0 - best_alpha_minilm], top_k=10)
        for q in all_queries
    }
    all_models_preds["Hybrid_ScoreFusion_BGE"] = {
        q["query_id"]: score_fusion([pool_bm25[q["query_id"]], pool_bge[q["query_id"]]], weights=[best_alpha_bge, 1.0 - best_alpha_bge], top_k=10)
        for q in all_queries
    }

    # Cross-encoder on all 90 queries
    logger.info("Executing Cross-Encoder reranking across all 90 queries...")
    rerank_preds_all: Dict[str, List[Tuple[str, float]]] = {}
    for q in all_queries:
        qid = q["query_id"]
        top50 = score_fusion([pool_bm25[qid], pool_bge[qid]], weights=[best_alpha_bge, 1.0 - best_alpha_bge], top_k=50)
        cand_ids = [doc_id for doc_id, _ in top50]
        rerank_preds_all[qid] = reranker.rerank_doc_lookup(q["query"], cand_ids, doc_lookup, top_k=10)
    all_models_preds["Hybrid_BGE_CrossEncoder"] = rerank_preds_all

    overall_eval_results: Dict[str, Any] = {}
    for model_name, preds in all_models_preds.items():
        summary = aggregate_query_metrics(all_queries, preds, qrels)
        overall_eval_results[model_name] = summary

    overall_significance = {
        "Hybrid_RRF_BGE_vs_BM25": paired_bootstrap_test(
            overall_eval_results["Hybrid_RRF_BGE"]["per_query_ndcg"],
            overall_eval_results["BM25"]["per_query_ndcg"],
        ),
        "Hybrid_ScoreFusion_BGE_vs_BM25": paired_bootstrap_test(
            overall_eval_results["Hybrid_ScoreFusion_BGE"]["per_query_ndcg"],
            overall_eval_results["BM25"]["per_query_ndcg"],
        ),
        "Hybrid_ScoreFusion_BGE_vs_BGE": paired_bootstrap_test(
            overall_eval_results["Hybrid_ScoreFusion_BGE"]["per_query_ndcg"],
            overall_eval_results["BGE"]["per_query_ndcg"],
        ),
        "Hybrid_BGE_CrossEncoder_vs_Hybrid_ScoreFusion_BGE": paired_bootstrap_test(
            overall_eval_results["Hybrid_BGE_CrossEncoder"]["per_query_ndcg"],
            overall_eval_results["Hybrid_ScoreFusion_BGE"]["per_query_ndcg"],
        ),
        "Hybrid_BGE_CrossEncoder_vs_BM25": paired_bootstrap_test(
            overall_eval_results["Hybrid_BGE_CrossEncoder"]["per_query_ndcg"],
            overall_eval_results["BM25"]["per_query_ndcg"],
        ),
        "Hybrid_BGE_CrossEncoder_vs_BGE": paired_bootstrap_test(
            overall_eval_results["Hybrid_BGE_CrossEncoder"]["per_query_ndcg"],
            overall_eval_results["BGE"]["per_query_ndcg"],
        ),
    }

    # ---------------------------------------------------------
    # STEP 4: PLOTTING TUNING CURVES & COMPARISON
    # ---------------------------------------------------------
    logger.info("Generating tuning and comparison plots...")
    # Plot 1: Weight Tuning Curve (Dev vs Test)
    plt.figure(figsize=(9, 5.5), dpi=300)
    alphas = sorted(dev_alpha_results_bge.keys())
    dev_ndcg_bge = [dev_alpha_results_bge[a]["ndcg@10"] for a in alphas]
    test_ndcg_bge = [test_alpha_results_bge[a]["ndcg@10"] for a in alphas]
    dev_ndcg_m = [dev_alpha_results_minilm[a]["ndcg@10"] for a in alphas]
    test_ndcg_m = [test_alpha_results_minilm[a]["ndcg@10"] for a in alphas]

    plt.plot(alphas, dev_ndcg_bge, "o-", color="#1f77b4", linewidth=2.2, label=f"BGE + BM25 (Dev Split, opt α={best_alpha_bge})")
    plt.plot(alphas, test_ndcg_bge, "s--", color="#17becf", linewidth=1.8, label=f"BGE + BM25 (Held-Out Test)")
    plt.plot(alphas, dev_ndcg_m, "^-", color="#ff7f0e", linewidth=2.0, label=f"MiniLM + BM25 (Dev Split, opt α={best_alpha_minilm})")
    plt.plot(alphas, test_ndcg_m, "d--", color="#bcbd22", linewidth=1.8, label="MiniLM + BM25 (Held-Out Test)")

    plt.axvline(x=best_alpha_bge, color="#d62728", linestyle=":", label=f"Selected α* = {best_alpha_bge}")
    plt.title("Hybrid Score Fusion Weight Tuning (Dev vs. Held-Out Test)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Lexical Weight (α) in Score Fusion  [1-α is Dense Semantic Weight]", fontsize=11)
    plt.ylabel("nDCG@10", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    plt.tight_layout()
    tuning_plot_path = figures_dir / "hybrid_weight_tuning.png"
    plt.savefig(tuning_plot_path)
    plt.close()
    logger.info("Saved tuning curve to %s", tuning_plot_path)

    # Plot 2: Model Performance Comparison Bar Chart
    plt.figure(figsize=(10, 5.5), dpi=300)
    display_models = [
        "TF-IDF",
        "BM25",
        "MiniLM",
        "BGE",
        "Hybrid_RRF_BGE",
        "Hybrid_ScoreFusion_BGE",
        "Hybrid_BGE_CrossEncoder",
    ]
    display_labels = [
        "TF-IDF",
        "BM25",
        "MiniLM",
        "BGE",
        "RRF (BGE+BM25)",
        "ScoreFusion (BGE+BM25)",
        "Cross-Encoder Reranked",
    ]
    test_scores = [test_eval_results[m]["ndcg_at_10"] for m in display_models]
    overall_scores = [overall_eval_results[m]["ndcg_at_10"] for m in display_models]

    x = np.arange(len(display_models))
    width = 0.38

    bars1 = plt.bar(x - width/2, test_scores, width, label="Held-Out Test Set (N=45)", color="#4C72B0")
    bars2 = plt.bar(x + width/2, overall_scores, width, label="Overall Benchmark (N=90)", color="#55A868")

    plt.title("Retriever Comparison: Lexical vs Dense vs Hybrid vs Cross-Encoder", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("nDCG@10", fontsize=11)
    plt.xticks(x, display_labels, rotation=20, ha="right", fontsize=9.5)
    plt.ylim(0, 0.85)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)

    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    comparison_plot_path = figures_dir / "hybrid_vs_baselines.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    logger.info("Saved comparison bar chart to %s", comparison_plot_path)

    # ---------------------------------------------------------
    # STEP 5: SAVE COMPLETE TUNING REPORT TO JSON
    # ---------------------------------------------------------
    tuning_report = {
        "tuning_metadata": {
            "dev_queries_count": len(dev_queries),
            "test_queries_count": len(test_queries),
            "total_queries_count": len(all_queries),
            "dev_ratio": 0.5,
            "seed": 42,
        },
        "dev_tuning": {
            "best_alpha_minilm": best_alpha_minilm,
            "best_alpha_bge": best_alpha_bge,
            "best_k_rrf_minilm": best_k_rrf_minilm,
            "best_k_rrf_bge": best_k_rrf_bge,
            "alpha_sweep_bge_dev": dev_alpha_results_bge,
            "alpha_sweep_bge_test": test_alpha_results_bge,
            "k_rrf_sweep_bge_dev": dev_rrf_results_bge,
        },
        "test_split_results": {
            m: {k: v for k, v in res.items() if k not in ("per_query_ndcg", "per_query_map")}
            for m, res in test_eval_results.items()
        },
        "test_statistical_significance": test_significance,
        "overall_benchmark_results": {
            m: {k: v for k, v in res.items() if k not in ("per_query_ndcg", "per_query_map")}
            for m, res in overall_eval_results.items()
        },
        "overall_statistical_significance": overall_significance,
    }

    report_path = results_dir / "hybrid_tuning.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, indent=2)
    logger.info("Saved complete hybrid tuning report to %s", report_path)

    # Print summary tables to stdout
    print("\n" + "=" * 90)
    print(" " * 22 + "HELD-OUT TEST SET EVALUATION REPORT (N=45)")
    print("=" * 90)
    print(f"{'Retriever':<26} | {'nDCG@10':<9} | {'95% CI':<16} | {'MAP':<8} | {'MRR':<8} | {'P@10':<8} | {'R@10':<8}")
    print("-" * 90)
    for m in display_models:
        o = test_eval_results[m]
        ci_str = f"[{o['ndcg_95_ci'][0]:.3f}, {o['ndcg_95_ci'][1]:.3f}]"
        print(f"{m:<26} | {o['ndcg_at_10']:<9.4f} | {ci_str:<16} | {o['map']:<8.4f} | {o['mrr']:<8.4f} | {o['precision_at_10']:<8.4f} | {o['recall_at_10']:<8.4f}")
    print("-" * 90)
    print("TEST SET SIGNIFICANCE (Paired Bootstrap, B=1000):")
    for comp, stats_data in test_significance.items():
        sig_mark = "*** (p<0.01)" if stats_data["statistically_significant_01"] else ("* (p<0.05)" if stats_data["statistically_significant_05"] else "n.s.")
        print(f"  {comp:<45}: diff={stats_data['mean_diff']:+.4f}, p={stats_data['bootstrap_p_value']:.4f} {sig_mark}")
    print("=" * 90 + "\n")

    return tuning_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retriever tuning and test evaluation")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument("--max-docs", type=int, default=5000)
    args = parser.parse_args()

    run_hybrid_tuning(config_path=args.config, max_corpus_docs=args.max_docs)


if __name__ == "__main__":
    main()
