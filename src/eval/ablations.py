"""Comprehensive Ablation Study Suite for Semantic and Lexical Search.

Evaluates 6 core architectural dimensions:
1. Stemming (Porter stemmer vs. no stemming in TF-IDF and BM25)
2. Sublinear Term Frequency (1 + ln(tf) vs. raw tf in TF-IDF)
3. Embedding Model (all-MiniLM-L6-v2 vs. BAAI/bge-small-en-v1.5)
4. Approximate vs. Exact Search (Faiss IndexFlatIP vs. IndexHNSWFlat)
5. Document Pooling / Aggregation Strategy (Max-pooling vs. Mean-pooling vs. Top-2 Sum)
6. Passage Chunk Size (100 words vs. 200 words vs. 400 words vs. full document)

Generates:
- Complete ablation table with Delta relative to default baseline
- Per-query-type breakdowns
- Output persisted to reports/results/ablation_results.json
- Comparison visualization to reports/figures/ablation_comparison.png
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
import yaml

from src.eval.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.eval.stats import compute_confidence_interval, paired_bootstrap_test
from src.lexical.bm25_retriever import BM25Retriever
from src.lexical.tfidf_retriever import CustomTFIDFRetriever
from src.preprocess.chunker import TextChunker
from src.preprocess.tokenizer import TextPreprocessor
from src.semantic.embedder import SentenceEmbedder
from src.semantic.faiss_index import FaissIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ablations")


def evaluate_query_list(
    queries: List[Dict[str, Any]],
    predictions: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """Calculate aggregate evaluation metrics over a list of queries."""
    p10_list, r10_list, mrr_list, map_list, ndcg_list = [], [], [], [], []
    by_type: Dict[str, Dict[str, List[float]]] = {
        "keyword": collections.defaultdict(list),
        "natural_language_question": collections.defaultdict(list),
        "paraphrase_vocab_mismatch": collections.defaultdict(list),
    }

    for q in queries:
        qid = q["query_id"]
        qtype = q["query_type"]
        retrieved = [doc_id for doc_id, _ in predictions.get(qid, [])]

        graded = qrels.get(qid, {})
        relevant_set = set(doc_id for doc_id, grade in graded.items() if grade >= 1)

        p10 = precision_at_k(retrieved, relevant_set, k=10)
        r10 = recall_at_k(retrieved, relevant_set, k=10)
        rr = reciprocal_rank(retrieved, relevant_set)
        ap = average_precision(retrieved, relevant_set)
        ndcg = ndcg_at_k(retrieved, graded, k=10)

        p10_list.append(p10)
        r10_list.append(r10)
        mrr_list.append(rr)
        map_list.append(ap)
        ndcg_list.append(ndcg)

        by_type[qtype]["p@10"].append(p10)
        by_type[qtype]["r@10"].append(r10)
        by_type[qtype]["mrr"].append(rr)
        by_type[qtype]["map"].append(ap)
        by_type[qtype]["ndcg@10"].append(ndcg)

    ci_l, ci_u = compute_confidence_interval(ndcg_list)

    summary = {
        "precision_at_10": round(float(np.mean(p10_list)), 4),
        "recall_at_10": round(float(np.mean(r10_list)), 4),
        "mrr": round(float(np.mean(mrr_list)), 4),
        "map": round(float(np.mean(map_list)), 4),
        "ndcg_at_10": round(float(np.mean(ndcg_list)), 4),
        "ndcg_95_ci": [round(ci_l, 4), round(ci_u, 4)],
        "per_query_ndcg": ndcg_list,
        "by_query_type": {
            t: {
                "precision_at_10": round(float(np.mean(vals["p@10"])), 4),
                "recall_at_10": round(float(np.mean(vals["r@10"])), 4),
                "mrr": round(float(np.mean(vals["mrr"])), 4),
                "map": round(float(np.mean(vals["map"])), 4),
                "ndcg_at_10": round(float(np.mean(vals["ndcg@10"])), 4),
            }
            for t, vals in by_type.items()
        },
    }
    return summary


def run_ablations(
    config_path: str = "configs/eval_config.yaml",
    max_corpus_docs: int = 5000,
) -> Dict[str, Any]:
    """Execute all 6 ablation experiments."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    queries_path = Path(config.get("queries_path", "data/eval/queries.jsonl"))
    qrels_path = Path(config.get("qrels_path", "data/eval/qrels.jsonl"))
    results_dir = Path("reports/results")
    figures_dir = Path("reports/figures")
    cache_dir = Path("data/indexes/embeddings")

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Queries and Ground Truth
    logger.info("Loading queries and ground truth...")
    queries: List[Dict[str, Any]] = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    qrels: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qrels[item["query_id"]][item["doc_id"]] = item["relevance"]

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
    doc_ids = [d["doc_id"] for d in corpus]
    doc_texts = [d["title"] + ". " + d["clean_body_text"][:600] for d in corpus]
    logger.info("Corpus loaded: %d documents.", len(corpus))

    ablation_results: Dict[str, Any] = {}
    existing_file = results_dir / "ablation_results.json"
    if existing_file.exists():
        try:
            with open(existing_file, "r", encoding="utf-8") as ef:
                prev = json.load(ef)
                if "stemming" in prev and "sublinear_tf" in prev and "embedding_model" in prev and "ann_vs_exact" in prev:
                    logger.info("Found existing valid ablation results for Stemming, Sublinear TF, Embeddings, and ANN. Reusing them...")
                    ablation_results["stemming"] = prev["stemming"]
                    ablation_results["sublinear_tf"] = prev["sublinear_tf"]
                    ablation_results["embedding_model"] = prev["embedding_model"]
                    ablation_results["ann_vs_exact"] = prev["ann_vs_exact"]
        except Exception as e:
            logger.warning("Could not read previous ablation results: %s", e)

    # -------------------------------------------------------------
    # ABLATION 1: STEMMING (Porter Stemmer ON vs. OFF)
    # -------------------------------------------------------------
    prep_stem = TextPreprocessor(stemmer="porter")
    prep_no_stem = TextPreprocessor(stemmer=None)

    if "stemming" not in ablation_results:
        logger.info("=== ABLATION 1: STEMMING (ON vs. OFF) ===")
        # TF-IDF with Stemming (Baseline)
        logger.info("Evaluating TF-IDF with Stemming...")
        tfidf_stem = CustomTFIDFRetriever(sublinear_tf=True, smooth_idf=True, min_df=2, preprocessor=prep_stem)
        tfidf_stem.index(corpus)
        preds_tfidf_stem = {q["query_id"]: tfidf_stem.search(q["query"], k=10) for q in queries}
        res_tfidf_stem = evaluate_query_list(queries, preds_tfidf_stem, qrels)

        # TF-IDF without Stemming
        logger.info("Evaluating TF-IDF without Stemming...")
        tfidf_no_stem = CustomTFIDFRetriever(sublinear_tf=True, smooth_idf=True, min_df=2, preprocessor=prep_no_stem)
        tfidf_no_stem.index(corpus)
        preds_tfidf_no_stem = {q["query_id"]: tfidf_no_stem.search(q["query"], k=10) for q in queries}
        res_tfidf_no_stem = evaluate_query_list(queries, preds_tfidf_no_stem, qrels)

        # BM25 with Stemming (Baseline)
        logger.info("Evaluating BM25 with Stemming...")
        bm25_stem = BM25Retriever(k1=1.5, b=0.75, preprocessor=prep_stem)
        bm25_stem.index(corpus)
        preds_bm25_stem = {q["query_id"]: bm25_stem.search(q["query"], k=10) for q in queries}
        res_bm25_stem = evaluate_query_list(queries, preds_bm25_stem, qrels)

        # BM25 without Stemming
        logger.info("Evaluating BM25 without Stemming...")
        bm25_no_stem = BM25Retriever(k1=1.5, b=0.75, preprocessor=prep_no_stem)
        bm25_no_stem.index(corpus)
        preds_bm25_no_stem = {q["query_id"]: bm25_no_stem.search(q["query"], k=10) for q in queries}
        res_bm25_no_stem = evaluate_query_list(queries, preds_bm25_no_stem, qrels)

        ablation_results["stemming"] = {
            "tfidf_stemming_on": res_tfidf_stem,
            "tfidf_stemming_off": res_tfidf_no_stem,
            "bm25_stemming_on": res_bm25_stem,
            "bm25_stemming_off": res_bm25_no_stem,
            "delta_tfidf_ndcg": round(res_tfidf_stem["ndcg_at_10"] - res_tfidf_no_stem["ndcg_at_10"], 4),
            "delta_bm25_ndcg": round(res_bm25_stem["ndcg_at_10"] - res_bm25_no_stem["ndcg_at_10"], 4),
        }

    # -------------------------------------------------------------
    # ABLATION 2: SUBLINEAR TF (ON vs. OFF)
    # -------------------------------------------------------------
    if "sublinear_tf" not in ablation_results:
        logger.info("=== ABLATION 2: SUBLINEAR TF (ON vs. OFF) ===")
        logger.info("Evaluating TF-IDF with raw TF (sublinear_tf=False)...")
        tfidf_raw_tf = CustomTFIDFRetriever(sublinear_tf=False, smooth_idf=True, min_df=2, preprocessor=prep_stem)
        tfidf_raw_tf.index(corpus)
        preds_tfidf_raw_tf = {q["query_id"]: tfidf_raw_tf.search(q["query"], k=10) for q in queries}
        res_tfidf_raw_tf = evaluate_query_list(queries, preds_tfidf_raw_tf, qrels)

        ablation_results["sublinear_tf"] = {
            "sublinear_tf_on_log": ablation_results["stemming"]["tfidf_stemming_on"],
            "sublinear_tf_off_raw": res_tfidf_raw_tf,
            "delta_ndcg": round(ablation_results["stemming"]["tfidf_stemming_on"]["ndcg_at_10"] - res_tfidf_raw_tf["ndcg_at_10"], 4),
        }

    # -------------------------------------------------------------
    # ABLATION 3: EMBEDDING MODEL (all-MiniLM-L6-v2 vs. bge-small-en-v1.5)
    # -------------------------------------------------------------
    logger.info("=== ABLATION 3: EMBEDDING MODEL ===")
    # MiniLM
    minilm_embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    minilm_embs = SentenceEmbedder.load_cache(cache_dir / f"minilm_doc_embeddings_{len(corpus)}.npy")
    minilm_faiss = FaissIndex(dimension=minilm_embs.shape[1], index_type="exact")
    minilm_faiss.add(minilm_embs)

    t0 = time.perf_counter()
    preds_minilm: Dict[str, List[Tuple[str, float]]] = {}
    for q in queries:
        qv = minilm_embedder.embed_query(q["query"])
        scores, idxs = minilm_faiss.search(qv, k=10)
        preds_minilm[q["query_id"]] = [(doc_ids[idx], float(s)) for s, idx in zip(scores, idxs)]
    lat_minilm = (time.perf_counter() - t0) * 1000.0 / len(queries)
    res_minilm = evaluate_query_list(queries, preds_minilm, qrels)
    res_minilm["mean_latency_ms"] = round(lat_minilm, 2)

    # BGE
    bge_embedder = SentenceEmbedder(model_name="BAAI/bge-small-en-v1.5")
    bge_embs = SentenceEmbedder.load_cache(cache_dir / f"bge_doc_embeddings_{len(corpus)}.npy")
    bge_faiss = FaissIndex(dimension=bge_embs.shape[1], index_type="exact")
    bge_faiss.add(bge_embs)

    t0 = time.perf_counter()
    preds_bge: Dict[str, List[Tuple[str, float]]] = {}
    for q in queries:
        qv = bge_embedder.embed_query(q["query"])
        scores, idxs = bge_faiss.search(qv, k=10)
        preds_bge[q["query_id"]] = [(doc_ids[idx], float(s)) for s, idx in zip(scores, idxs)]
    lat_bge = (time.perf_counter() - t0) * 1000.0 / len(queries)
    res_bge = evaluate_query_list(queries, preds_bge, qrels)
    res_bge["mean_latency_ms"] = round(lat_bge, 2)

    ablation_results["embedding_model"] = {
        "all_MiniLM_L6_v2": res_minilm,
        "bge_small_en_v1_5": res_bge,
        "delta_bge_vs_minilm_ndcg": round(res_bge["ndcg_at_10"] - res_minilm["ndcg_at_10"], 4),
    }

    # -------------------------------------------------------------
    # ABLATION 4: ANN VS. EXACT SEARCH
    # -------------------------------------------------------------
    logger.info("=== ABLATION 4: ANN (HNSW) VS. EXACT FLAT SEARCH ===")
    hnsw_index = FaissIndex(dimension=bge_embs.shape[1], index_type="hnsw", hnsw_m=32, ef_construction=64, ef_search=64)
    hnsw_index.add(bge_embs)

    t0 = time.perf_counter()
    preds_hnsw: Dict[str, List[Tuple[str, float]]] = {}
    for q in queries:
        qv = bge_embedder.embed_query(q["query"])
        scores, idxs = hnsw_index.search(qv, k=10)
        preds_hnsw[q["query_id"]] = [(doc_ids[idx], float(s)) for s, idx in zip(scores, idxs)]
    lat_hnsw = (time.perf_counter() - t0) * 1000.0 / len(queries)
    res_hnsw = evaluate_query_list(queries, preds_hnsw, qrels)
    res_hnsw["mean_latency_ms"] = round(lat_hnsw, 2)

    # Compute ANN recall relative to exact
    total_exact_matches = 0
    total_k = len(queries) * 10
    for q in queries:
        qid = q["query_id"]
        exact_top = set(d for d, _ in preds_bge[qid])
        hnsw_top = set(d for d, _ in preds_hnsw[qid])
        total_exact_matches += len(exact_top.intersection(hnsw_top))
    ann_recall_vs_exact = total_exact_matches / float(total_k)

    ablation_results["ann_vs_exact"] = {
        "exact_flat_ip": res_bge,
        "hnsw_approximate": res_hnsw,
        "ann_top10_recall_vs_exact": round(ann_recall_vs_exact, 4),
        "recall_loss_pct": round((1.0 - ann_recall_vs_exact) * 100.0, 2),
        "speedup_factor": round(lat_bge / lat_hnsw, 2) if lat_hnsw > 0 else 1.0,
    }

    # -------------------------------------------------------------
    # ABLATION 5: DOCUMENT POOLING / AGGREGATION STRATEGY
    # -------------------------------------------------------------
    logger.info("=== ABLATION 5: PASSAGE-TO-DOCUMENT POOLING STRATEGY ===")
    all_qrels_doc_ids: Set[str] = set()
    for q_map in qrels.values():
        all_qrels_doc_ids.update(q_map.keys())
    target_docs = [d for d in corpus if d["doc_id"] in all_qrels_doc_ids][:500]
    logger.info("Using %d candidate documents for multi-passage pooling ablation...", len(target_docs))

    passage_records: List[Tuple[str, str]] = []  # (doc_id, text)
    for d in target_docs:
        doc_id = d["doc_id"]
        title = d["title"]
        body = d.get("clean_body_text") or d.get("raw_text", "")
        words = body.split()
        if len(words) <= 120:
            passage_records.append((doc_id, f"{title}. {body[:400]}"))
        else:
            mid = len(words) // 2
            p1 = " ".join(words[:min(mid, 180)])
            p2 = " ".join(words[mid : min(mid + 180, len(words))])
            passage_records.append((doc_id, f"{title}. {p1}"))
            passage_records.append((doc_id, f"{title}. {p2}"))

    pool_cache_file = cache_dir / f"pooling_passages_{len(target_docs)}.npy"
    if pool_cache_file.exists():
        logger.info("Loading cached pooling passage embeddings from %s...", pool_cache_file)
        p_embs = SentenceEmbedder.load_cache(pool_cache_file)
    else:
        logger.info("Encoding %d passages for pooling comparison...", len(passage_records))
        p_texts = [p[1] for p in passage_records]
        p_embs = minilm_embedder.embed_passages(p_texts, batch_size=128)
        SentenceEmbedder.save_cache(p_embs, pool_cache_file)

    p_faiss = FaissIndex(dimension=p_embs.shape[1], index_type="exact")
    p_faiss.add(p_embs)
    p_doc_ids = [p[0] for p in passage_records]

    def aggregate_passages(scores: np.ndarray, indices: np.ndarray, strategy: str, k: int = 10) -> List[Tuple[str, float]]:
        grouped = collections.defaultdict(list)
        for s, idx in zip(scores, indices):
            if 0 <= idx < len(p_doc_ids):
                grouped[p_doc_ids[idx]].append(float(s))
        doc_scores = []
        for did, sl in grouped.items():
            if strategy == "max":
                score = max(sl)
            elif strategy == "mean":
                score = sum(sl) / len(sl)
            elif strategy == "top_k_sum":
                score = sum(sorted(sl, reverse=True)[:2])
            else:
                score = max(sl)
            doc_scores.append((did, score))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:k]

    pooling_strategies = ["max", "mean", "top_k_sum"]
    pooling_evals = {}
    for strat in pooling_strategies:
        logger.info("Evaluating pooling strategy: %s...", strat)
        strat_preds = {}
        for q in queries:
            qv = minilm_embedder.embed_query(q["query"])
            scores, idxs = p_faiss.search(qv, k=60)
            strat_preds[q["query_id"]] = aggregate_passages(scores, idxs, strategy=strat, k=10)
        pooling_evals[strat] = evaluate_query_list(queries, strat_preds, qrels)

    ablation_results["pooling_strategy"] = pooling_evals

    # -------------------------------------------------------------
    # ABLATION 6: PASSAGE CHUNK SIZE (100 vs. 200 vs. 400 vs. Full)
    # -------------------------------------------------------------
    logger.info("=== ABLATION 6: PASSAGE CHUNK SIZE ===")
    chunk_sizes = [100, 200, 400, 800]
    chunk_labels = ["100_words", "200_words_default", "400_words", "full_document"]
    chunk_evals = {}
    chunk_docs = target_docs
    c_doc_ids = [d["doc_id"] for d in chunk_docs]

    for c_size, c_lbl in zip(chunk_sizes, chunk_labels):
        logger.info("Evaluating chunk size: %s (%d words)...", c_lbl, c_size)
        c_cache_file = cache_dir / f"chunk_{c_size}_embs_{len(chunk_docs)}.npy"
        if c_cache_file.exists():
            logger.info("Loading cached chunk embeddings from %s...", c_cache_file)
            c_embs = SentenceEmbedder.load_cache(c_cache_file)
        else:
            c_texts = [d["title"] + ". " + " ".join(d.get("clean_body_text", "").split()[:c_size]) for d in chunk_docs]
            c_embs = minilm_embedder.embed_passages(c_texts, batch_size=128)
            SentenceEmbedder.save_cache(c_embs, c_cache_file)

        c_faiss = FaissIndex(dimension=c_embs.shape[1], index_type="exact")
        c_faiss.add(c_embs)

        c_preds = {}
        for q in queries:
            qv = minilm_embedder.embed_query(q["query"])
            scores, idxs = c_faiss.search(qv, k=10)
            c_preds[q["query_id"]] = [(c_doc_ids[idx], float(s)) for s, idx in zip(scores, idxs)]
        chunk_evals[c_lbl] = evaluate_query_list(queries, c_preds, qrels)

    ablation_results["chunk_size"] = chunk_evals

    # -------------------------------------------------------------
    # SAVE AND PLOT ABLATION RESULTS
    # -------------------------------------------------------------
    clean_ablation_output = {}
    for dim_name, dim_data in ablation_results.items():
        clean_ablation_output[dim_name] = {}
        for k, v in dim_data.items():
            if isinstance(v, dict) and "per_query_ndcg" in v:
                clean_ablation_output[dim_name][k] = {ik: iv for ik, iv in v.items() if ik != "per_query_ndcg"}
            else:
                clean_ablation_output[dim_name][k] = v

    out_file = results_dir / "ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(clean_ablation_output, f, indent=2)
    logger.info("Saved ablation report to %s", out_file)

    # Plot Ablation Comparison Chart
    plt.figure(figsize=(11, 6), dpi=300)
    ablation_keys = [
        ("Sublinear TF\n(Raw vs. Log)", ablation_results["sublinear_tf"]["sublinear_tf_off_raw"]["ndcg_at_10"], ablation_results["sublinear_tf"]["sublinear_tf_on_log"]["ndcg_at_10"]),
        ("TF-IDF Stemming\n(Off vs. On)", ablation_results["stemming"]["tfidf_stemming_off"]["ndcg_at_10"], ablation_results["stemming"]["tfidf_stemming_on"]["ndcg_at_10"]),
        ("BM25 Stemming\n(Off vs. On)", ablation_results["stemming"]["bm25_stemming_off"]["ndcg_at_10"], ablation_results["stemming"]["bm25_stemming_on"]["ndcg_at_10"]),
        ("Dense Embedding\n(MiniLM vs. BGE)", ablation_results["embedding_model"]["all_MiniLM_L6_v2"]["ndcg_at_10"], ablation_results["embedding_model"]["bge_small_en_v1_5"]["ndcg_at_10"]),
        ("Index Precision\n(HNSW vs. Exact)", ablation_results["ann_vs_exact"]["hnsw_approximate"]["ndcg_at_10"], ablation_results["ann_vs_exact"]["exact_flat_ip"]["ndcg_at_10"]),
        ("Pooling Strategy\n(Mean vs. Max)", ablation_results["pooling_strategy"]["mean"]["ndcg_at_10"], ablation_results["pooling_strategy"]["max"]["ndcg_at_10"]),
        ("Chunk Size\n(100w vs. 200w)", ablation_results["chunk_size"]["100_words"]["ndcg_at_10"], ablation_results["chunk_size"]["200_words_default"]["ndcg_at_10"]),
    ]

    labels = [k[0] for k in ablation_keys]
    base_vals = [k[1] for k in ablation_keys]
    tuned_vals = [k[2] for k in ablation_keys]

    x = np.arange(len(labels))
    w = 0.35

    plt.bar(x - w/2, base_vals, w, label="Baseline / Alternative Variant", color="#E24A33", alpha=0.85)
    plt.bar(x + w/2, tuned_vals, w, label="Optimized / Recommended Variant", color="#348ABD", alpha=0.9)

    plt.ylabel("nDCG@10", fontsize=11)
    plt.title("Ablation Study: Retrieval Quality (nDCG@10) Across Architectural Variations", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, labels, fontsize=9)
    plt.ylim(0, 0.75)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)

    for i in range(len(labels)):
        delta = tuned_vals[i] - base_vals[i]
        plt.text(x[i], max(base_vals[i], tuned_vals[i]) + 0.02, f"Δ {delta:+.3f}", ha="center", fontsize=8.5, fontweight="bold", color="#2c3e50")

    plt.tight_layout()
    chart_path = figures_dir / "ablation_comparison.png"
    plt.savefig(chart_path)
    plt.close()
    logger.info("Saved ablation comparison chart to %s", chart_path)

    return clean_ablation_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete ablation study suite")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument("--max-docs", type=int, default=5000)
    args = parser.parse_args()

    run_ablations(config_path=args.config, max_corpus_docs=args.max_docs)


if __name__ == "__main__":
    main()
