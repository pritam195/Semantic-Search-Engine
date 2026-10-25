"""Benchmarking script for lexical retrievers (Custom TF-IDF vs BM25).

Measures:
- Indexing time (seconds)
- Persisted index size on disk (MB)
- Query retrieval latency across 100 queries (mean, median, p95, p99 in ms)
- Query throughput (Queries Per Second - QPS)
- Programmatic output saved to reports/results/lexical_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from tqdm import tqdm

from src.lexical.bm25_retriever import BM25Retriever
from src.lexical.tfidf_retriever import CustomTFIDFRetriever
from src.preprocess.tokenizer import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("lexical_benchmark")

# 100 domain-specific queries covering CS, ML, and Statistics
DEFAULT_BENCHMARK_QUERIES: List[str] = [
    "machine learning algorithms",
    "deep neural networks backpropagation",
    "gradient descent optimization methods",
    "support vector machines kernel trick",
    "principal component analysis dimensionality reduction",
    "natural language processing word embeddings",
    "bayesian inference prior and posterior distributions",
    "decision trees and random forest classifier",
    "k-means clustering unsupervised learning",
    "reinforcement learning markov decision process",
    "linear regression ordinary least squares",
    "stochastic gradient descent learning rate",
    "overfitting and regularization techniques L1 L2",
    "cross validation and hyperparameter tuning",
    "convolutional neural networks computer vision",
    "recurrent neural networks LSTM sequence modeling",
    "transformer architecture self attention mechanism",
    "loss functions cross entropy mean squared error",
    "activation functions relu sigmoid tanh",
    "matrix factorization singular value decomposition",
    "logistic regression binary classification",
    "generative adversarial networks generator discriminator",
    "variational autoencoders latent space representation",
    "ensemble learning boosting and bagging",
    "gradient boosting xgboost lightgbm",
    "naive bayes classifier conditional probability",
    "k-nearest neighbors distance metrics",
    "information retrieval inverted index tf-idf",
    "cosine similarity vector space model",
    "markov chain monte carlo sampling",
    "maximum likelihood estimation parameter inference",
    "expectation maximization Gaussian mixture models",
    "hypothesis testing p-value significance",
    "central limit theorem normal distribution",
    "confidence intervals bootstrap resampling",
    "eigenvalues and eigenvectors linear algebra",
    "convex optimization duality Lagrange multipliers",
    "graph algorithms shortest path Dijkstra",
    "breadth first search depth first search",
    "computational complexity NP-completeness",
    "sorting algorithms quicksort mergesort",
    "hash tables collision resolution",
    "binary search trees balancing AVL red-black",
    "operating systems process scheduling concurrency",
    "relational database normalization SQL indexing",
    "data structures stack queue linked list",
    "dynamic programming memoization optimal substructure",
    "graph neural networks node classification",
    "transfer learning pre-trained models fine-tuning",
    "attention is all you need multi-head attention",
    "batch normalization internal covariate shift",
    "dropout regularization neural network training",
    "learning rate schedulers cosine annealing warmup",
    "object detection yolo r-cnn bounding boxes",
    "semantic segmentation u-net mask r-cnn",
    "speech recognition acoustic model phonemes",
    "recommendation systems collaborative filtering matrix factorization",
    "time series forecasting ARIMA exponential smoothing",
    "anomaly detection isolation forest one-class SVM",
    "semi-supervised learning pseudo labeling",
    "self-supervised learning contrastive learning SimCLR",
    "few-shot learning prototypical networks",
    "model interpretability SHAP LIME feature importance",
    "fairness in machine learning bias mitigation",
    "differential privacy epsilon delta guarantee",
    "federated learning decentralized model training",
    "knowledge distillation student teacher network",
    "quantization and pruning model compression",
    "hardware acceleration GPU tensor cores TPU",
    "distributed training data parallelism model parallelism",
    "hyperparameter optimization Bayesian optimization Optuna",
    "ROC curve AUC metric classification evaluation",
    "precision recall f1-score confusion matrix",
    "mean average precision ranking evaluation",
    "normalized discounted cumulative gain NDCG",
    "perplexity language model evaluation",
    "word2vec skip-gram continuous bag of words",
    "BERT bidirectional encoder representations",
    "GPT autoregressive language generation",
    "prompt engineering few-shot prompting in-context learning",
    "retrieval augmented generation external knowledge",
    "vector database similarity search approximate nearest neighbor",
    "HNSW hierarchical navigable small world graphs",
    "Faiss library billion scale similarity search",
    "BM25 ranking function Okapi BM25",
    "edit distance Levenshtein distance string similarity",
    "stemming and lemmatization text normalization",
    "n-gram language models Markov models",
    "hidden Markov models Viterbi algorithm",
    "conditional random fields sequence labeling",
    "topic modeling Latent Dirichlet Allocation",
    "sentiment analysis text classification",
    "named entity recognition sequence tagging",
    "coreference resolution natural language understanding",
    "dependency parsing syntactic structure",
    "machine translation sequence to sequence attention",
    "question answering reading comprehension",
    "information extraction relation extraction",
    "knowledge graphs entity linking RDF triples",
    "curse of dimensionality high dimensional data",
]


def get_dir_size_mb(path: Path) -> float:
    """Calculate total size of directory contents in megabytes."""
    total_bytes = 0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)


def run_benchmark(
    config_path: str = "configs/index_config.yaml",
    max_corpus_records: int | None = None,
) -> Dict[str, Any]:
    """Execute end-to-end indexing and retrieval benchmark for Custom TF-IDF and BM25."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    corpus_path = Path(config["corpus_path"])
    indexes_dir = Path(config["indexes_dir"])
    tfidf_dir = indexes_dir / "tfidf"
    bm25_dir = indexes_dir / "bm25"
    tfidf_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir.mkdir(parents=True, exist_ok=True)

    tfidf_cfg = config.get("tfidf", {})
    bm25_cfg = config.get("bm25", {})
    bench_cfg = config.get("benchmark", {})
    top_k = bench_cfg.get("top_k", 10)
    output_json = Path(bench_cfg.get("results_output", "reports/results/lexical_benchmark.json"))
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Processed Corpus
    logger.info("Loading processed corpus from %s...", corpus_path)
    corpus: List[Dict[str, Any]] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
            if max_corpus_records and len(corpus) >= max_corpus_records:
                break
    n_records = len(corpus)
    logger.info("Loaded %d corpus records for indexing.", n_records)

    preprocessor = TextPreprocessor(stemmer="porter", stopwords="english")
    queries = DEFAULT_BENCHMARK_QUERIES[: bench_cfg.get("num_queries", 100)]

    # 2. Benchmark Custom TF-IDF
    logger.info("--- Benchmarking Custom TF-IDF ---")
    t0 = time.perf_counter()
    tfidf_retriever = CustomTFIDFRetriever(
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        smooth_idf=tfidf_cfg.get("smooth_idf", True),
        norm=tfidf_cfg.get("norm", "l2"),
        min_df=tfidf_cfg.get("min_df", 3),
        max_df=tfidf_cfg.get("max_df", 0.85),
        max_features=tfidf_cfg.get("max_features", 60000),
        preprocessor=preprocessor,
    )
    tfidf_retriever.index(corpus)
    tfidf_index_time = time.perf_counter() - t0

    tfidf_retriever.save(tfidf_dir)
    tfidf_disk_size_mb = get_dir_size_mb(tfidf_dir)

    # Latency benchmark
    tfidf_latencies: List[float] = []
    for q in queries:
        t_start = time.perf_counter()
        _ = tfidf_retriever.search(q, k=top_k)
        t_lat_ms = (time.perf_counter() - t_start) * 1000.0
        tfidf_latencies.append(t_lat_ms)

    tfidf_stats = {
        "indexing_time_sec": round(tfidf_index_time, 3),
        "index_size_mb": round(tfidf_disk_size_mb, 2),
        "vocabulary_size": len(tfidf_retriever.vocabulary),
        "non_zero_elements": int(tfidf_retriever.doc_matrix.nnz) if tfidf_retriever.doc_matrix is not None else 0,
        "mean_latency_ms": round(float(np.mean(tfidf_latencies)), 3),
        "median_latency_ms": round(float(np.median(tfidf_latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(tfidf_latencies, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(tfidf_latencies, 99)), 3),
        "qps": round(1000.0 / float(np.mean(tfidf_latencies)), 1),
    }

    # 3. Benchmark BM25
    logger.info("--- Benchmarking BM25 (rank_bm25) ---")
    t0 = time.perf_counter()
    bm25_retriever = BM25Retriever(
        k1=bm25_cfg.get("k1", 1.5),
        b=bm25_cfg.get("b", 0.75),
        epsilon=bm25_cfg.get("epsilon", 0.25),
        preprocessor=preprocessor,
    )
    bm25_retriever.index(corpus)
    bm25_index_time = time.perf_counter() - t0

    bm25_retriever.save(bm25_dir)
    bm25_disk_size_mb = get_dir_size_mb(bm25_dir)

    # Latency benchmark
    bm25_latencies: List[float] = []
    for q in queries:
        t_start = time.perf_counter()
        _ = bm25_retriever.search(q, k=top_k)
        t_lat_ms = (time.perf_counter() - t_start) * 1000.0
        bm25_latencies.append(t_lat_ms)

    bm25_stats = {
        "indexing_time_sec": round(bm25_index_time, 3),
        "index_size_mb": round(bm25_disk_size_mb, 2),
        "mean_latency_ms": round(float(np.mean(bm25_latencies)), 3),
        "median_latency_ms": round(float(np.median(bm25_latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(bm25_latencies, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(bm25_latencies, 99)), 3),
        "qps": round(1000.0 / float(np.mean(bm25_latencies)), 1),
    }

    # Compile benchmark summary
    results: Dict[str, Any] = {
        "corpus_records_indexed": n_records,
        "num_queries_evaluated": len(queries),
        "top_k": top_k,
        "tfidf_custom": tfidf_stats,
        "bm25_baseline": bm25_stats,
    }

    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    logger.info("Saved lexical benchmark results to %s", output_json)

    # Print summary comparison table
    print("\n" + "=" * 65)
    print(" " * 16 + "LEXICAL RETRIEVAL BENCHMARK REPORT")
    print("=" * 65)
    print(f"Corpus records indexed: {n_records:,} passages")
    print(f"Benchmark queries:      {len(queries)} queries (Top-k = {top_k})")
    print("-" * 65)
    print(f"{'Metric':<28} | {'Custom TF-IDF':<16} | {'BM25':<16}")
    print("-" * 65)
    print(f"{'Indexing Time':<28} | {tfidf_stats['indexing_time_sec']:>12.2f} s   | {bm25_stats['indexing_time_sec']:>12.2f} s")
    print(f"{'Index Size on Disk':<28} | {tfidf_stats['index_size_mb']:>12.2f} MB  | {bm25_stats['index_size_mb']:>12.2f} MB")
    print(f"{'Mean Latency':<28} | {tfidf_stats['mean_latency_ms']:>12.2f} ms  | {bm25_stats['mean_latency_ms']:>12.2f} ms")
    print(f"{'Median Latency':<28} | {tfidf_stats['median_latency_ms']:>12.2f} ms  | {bm25_stats['median_latency_ms']:>12.2f} ms")
    print(f"{'p95 Latency':<28} | {tfidf_stats['p95_latency_ms']:>12.2f} ms  | {bm25_stats['p95_latency_ms']:>12.2f} ms")
    print(f"{'p99 Latency':<28} | {tfidf_stats['p99_latency_ms']:>12.2f} ms  | {bm25_stats['p99_latency_ms']:>12.2f} ms")
    print(f"{'Query Throughput (QPS)':<28} | {tfidf_stats['qps']:>12.1f} req/s| {bm25_stats['qps']:>12.1f} req/s")
    print("=" * 65 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark lexical retrievers (TF-IDF vs BM25)")
    parser.add_argument("--config", type=str, default="configs/index_config.yaml")
    parser.add_argument("--max-records", type=int, default=None, help="Limit corpus records for quick run")
    args = parser.parse_args()

    run_benchmark(config_path=args.config, max_corpus_records=args.max_records)


if __name__ == "__main__":
    main()
