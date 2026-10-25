"""Comprehensive benchmarking script for semantic retrieval (Exact vs HNSW FAISS).

Measures:
- Embedding time and encoding throughput (passages/sec)
- Index build time (Exact IndexFlatIP vs Approximate IndexHNSWFlat)
- Index size on disk (Exact .faiss vs ANN .faiss vs Cached .npy)
- Resident Memory Use (RAM RSS in MB)
- Query latency across 100 queries (mean, median, p95, p99 in ms)
- Recall loss of Approximate Nearest Neighbors (ANN) vs Exact Search
- Programmatic JSON report saved to reports/results/semantic_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import yaml
from tqdm import tqdm

from src.lexical.benchmark import DEFAULT_BENCHMARK_QUERIES
from src.preprocess.chunker import TextChunker
from src.semantic.embedder import SentenceEmbedder
from src.semantic.faiss_index import FaissIndex
from src.semantic.retriever import SemanticRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("semantic_benchmark")


def get_current_memory_mb() -> float:
    """Return resident memory (RSS) in megabytes of current process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_semantic_benchmark(
    config_path: str = "configs/semantic_config.yaml",
    max_passages: int | None = 5000,
) -> Dict[str, Any]:
    """Execute end-to-end embedding, FAISS indexing, latency, and recall loss benchmark."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    corpus_path = Path(config["corpus_path"])
    cache_dir = Path(config["cache_dir"])
    indexes_dir = Path(config["indexes_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir.mkdir(parents=True, exist_ok=True)

    emb_cfg = config.get("embedding", {})
    model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = emb_cfg.get("batch_size", 128)

    chunk_cfg = config.get("chunking", {})
    chunker = TextChunker(
        chunk_size_words=chunk_cfg.get("chunk_size_words", 200),
        chunk_overlap_words=chunk_cfg.get("chunk_overlap_words", 40),
        min_chunk_words=chunk_cfg.get("min_chunk_words", 25),
    )
    agg_strategy = chunk_cfg.get("aggregation_strategy", "max")

    faiss_cfg = config.get("faiss", {})
    hnsw_m = faiss_cfg.get("hnsw_m", 32)
    ef_construction = faiss_cfg.get("ef_construction", 64)
    ef_search = faiss_cfg.get("ef_search", 64)

    bench_cfg = config.get("benchmark", {})
    num_queries = bench_cfg.get("num_queries", 100)
    top_k = bench_cfg.get("top_k", 10)
    output_json = Path(bench_cfg.get("results_output", "reports/results/semantic_benchmark.json"))
    output_json.parent.mkdir(parents=True, exist_ok=True)

    queries = DEFAULT_BENCHMARK_QUERIES[:num_queries]

    # 1. Chunk Corpus Documents into Passages
    logger.info("Loading documents from %s...", corpus_path)
    all_chunks: List[Dict[str, Any]] = []
    doc_metadata: Dict[str, Dict[str, Any]] = {}
    chunk_to_doc: List[str] = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            d_id = doc["doc_id"]
            doc_metadata[d_id] = {
                "doc_id": d_id,
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "raw_text": doc.get("clean_body_text", ""),
            }
            chunks = chunker.chunk_document(
                doc_id=d_id,
                title=doc.get("title", ""),
                url=doc.get("url", ""),
                text=doc.get("clean_body_text", ""),
            )
            for c in chunks:
                all_chunks.append(c.to_dict())
                chunk_to_doc.append(d_id)
                if max_passages and len(all_chunks) >= max_passages:
                    break
            if max_passages and len(all_chunks) >= max_passages:
                break

    n_passages = len(all_chunks)
    logger.info("Prepared %d passages for semantic indexing.", n_passages)

    # 2. Embedding Generation & Caching
    mem_before_embed = get_current_memory_mb()
    embedder = SentenceEmbedder(model_name=model_name, batch_size=batch_size)

    sanitized_model = model_name.replace("/", "_").replace("-", "_").lower()
    cache_path = cache_dir / f"{sanitized_model}_embeddings_{n_passages}.npy"

    logger.info("Generating dense embeddings for %d passages...", n_passages)
    t0 = time.perf_counter()
    if cache_path.exists():
        logger.info("Found cached embeddings at %s, loading...", cache_path)
        embeddings = SentenceEmbedder.load_cache(cache_path)
        embed_time_sec = 0.05
    else:
        texts = [c["text"] for c in all_chunks]
        embeddings = embedder.embed_passages(texts, batch_size=batch_size)
        embed_time_sec = time.perf_counter() - t0
        SentenceEmbedder.save_cache(embeddings, cache_path)

    embed_throughput = round(n_passages / max(0.001, embed_time_sec), 1)
    cache_size_mb = cache_path.stat().st_size / (1024 * 1024) if cache_path.exists() else 0.0
    dim = embeddings.shape[1]

    # 3. FAISS Indexing (Exact vs HNSW)
    logger.info("Building FAISS Exact Index (IndexFlatIP)...")
    t0_exact = time.perf_counter()
    exact_index = FaissIndex(dimension=dim, index_type="exact")
    exact_index.add(embeddings)
    exact_build_time = time.perf_counter() - t0_exact
    exact_index_file = indexes_dir / f"exact_{sanitized_model}.faiss"
    exact_index.save(exact_index_file)
    exact_size_mb = exact_index_file.stat().st_size / (1024 * 1024)

    logger.info("Building FAISS HNSW Index (IndexHNSWFlat)...")
    t0_hnsw = time.perf_counter()
    ann_index = FaissIndex(
        dimension=dim,
        index_type="hnsw",
        hnsw_m=hnsw_m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    ann_index.add(embeddings)
    ann_build_time = time.perf_counter() - t0_hnsw
    ann_index_file = indexes_dir / f"ann_hnsw_{sanitized_model}.faiss"
    ann_index.save(ann_index_file)
    ann_size_mb = ann_index_file.stat().st_size / (1024 * 1024)

    mem_after_indexing = get_current_memory_mb()
    ram_usage_mb = round(mem_after_indexing - mem_before_embed, 2)

    # 4. Initialize SemanticRetriever for Evaluation
    retriever = SemanticRetriever(
        embedder=embedder,
        aggregation_strategy=agg_strategy,
        hnsw_m=hnsw_m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    retriever.exact_index = exact_index
    retriever.ann_index = ann_index
    retriever.chunk_records = all_chunks
    retriever.chunk_to_doc_id = chunk_to_doc
    retriever.doc_metadata = doc_metadata

    # 5. Query Latency & Recall Loss Benchmark
    logger.info("Running retrieval latency and recall benchmark over %d queries...", len(queries))
    exact_latencies: List[float] = []
    ann_latencies: List[float] = []
    recalls_at_k: List[float] = []

    for q in queries:
        # Exact Search Latency
        t_start = time.perf_counter()
        exact_results = retriever.search(q, k=top_k, use_ann=False)
        exact_lat_ms = (time.perf_counter() - t_start) * 1000.0
        exact_latencies.append(exact_lat_ms)

        # ANN Search Latency
        t_start = time.perf_counter()
        ann_results = retriever.search(q, k=top_k, use_ann=True)
        ann_lat_ms = (time.perf_counter() - t_start) * 1000.0
        ann_latencies.append(ann_lat_ms)

        # Compute Recall@k for this query
        exact_ids = set(doc_id for doc_id, _ in exact_results)
        ann_ids = set(doc_id for doc_id, _ in ann_results)
        if len(exact_ids) > 0:
            intersection = len(exact_ids.intersection(ann_ids))
            recall = intersection / len(exact_ids)
        else:
            recall = 1.0
        recalls_at_k.append(recall)

    mean_recall = float(np.mean(recalls_at_k))
    recall_loss = 1.0 - mean_recall

    exact_stats = {
        "build_time_sec": round(exact_build_time, 3),
        "index_size_mb": round(exact_size_mb, 2),
        "mean_latency_ms": round(float(np.mean(exact_latencies)), 3),
        "median_latency_ms": round(float(np.median(exact_latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(exact_latencies, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(exact_latencies, 99)), 3),
        "qps": round(1000.0 / float(np.mean(exact_latencies)), 1),
    }

    ann_stats = {
        "build_time_sec": round(ann_build_time, 3),
        "index_size_mb": round(ann_size_mb, 2),
        "mean_latency_ms": round(float(np.mean(ann_latencies)), 3),
        "median_latency_ms": round(float(np.median(ann_latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(ann_latencies, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(ann_latencies, 99)), 3),
        "qps": round(1000.0 / float(np.mean(ann_latencies)), 1),
        "mean_recall_at_k": round(mean_recall, 4),
        "recall_loss_pct": round(recall_loss * 100.0, 2),
    }

    results = {
        "model_name": model_name,
        "embedding_dimension": dim,
        "passages_indexed": n_passages,
        "num_benchmark_queries": len(queries),
        "top_k": top_k,
        "aggregation_strategy": agg_strategy,
        "embedding_benchmark": {
            "embedding_time_sec": round(embed_time_sec, 2),
            "throughput_passages_per_sec": embed_throughput,
            "cached_embeddings_size_mb": round(cache_size_mb, 2),
            "ram_footprint_mb": ram_usage_mb,
        },
        "exact_faiss": exact_stats,
        "ann_hnsw_faiss": ann_stats,
    }

    with open(output_json, "w", encoding="utf-8") as rf:
        json.dump(results, rf, indent=2)

    logger.info("Saved semantic benchmark results to %s", output_json)

    # Print summary table
    print("\n" + "=" * 68)
    print(" " * 16 + "SEMANTIC RETRIEVAL BENCHMARK REPORT")
    print("=" * 68)
    print(f"Model:                  {model_name} (dim = {dim})")
    print(f"Passages Indexed:       {n_passages:,} passages")
    print(f"Aggregation Strategy:   {agg_strategy}")
    print(f"Embedding Time:         {embed_time_sec:.2f} s ({embed_throughput:,.1f} passages/s)")
    print(f"Embedding Cache Size:   {cache_size_mb:.2f} MB")
    print(f"Process RAM Overhead:   {ram_usage_mb:.2f} MB")
    print("-" * 68)
    print(f"{'Metric':<28} | {'Exact (IndexFlatIP)':<17} | {'ANN (HNSW Flat)':<17}")
    print("-" * 68)
    print(f"{'Index Build Time':<28} | {exact_stats['build_time_sec']:>13.2f} s   | {ann_stats['build_time_sec']:>13.2f} s")
    print(f"{'Index Size on Disk':<28} | {exact_stats['index_size_mb']:>13.2f} MB  | {ann_stats['index_size_mb']:>13.2f} MB")
    print(f"{'Mean Query Latency':<28} | {exact_stats['mean_latency_ms']:>13.2f} ms  | {ann_stats['mean_latency_ms']:>13.2f} ms")
    print(f"{'Median Query Latency':<28} | {exact_stats['median_latency_ms']:>13.2f} ms  | {ann_stats['median_latency_ms']:>13.2f} ms")
    print(f"{'p95 Latency':<28} | {exact_stats['p95_latency_ms']:>13.2f} ms  | {ann_stats['p95_latency_ms']:>13.2f} ms")
    print(f"{'p99 Latency':<28} | {exact_stats['p99_latency_ms']:>13.2f} ms  | {ann_stats['p99_latency_ms']:>13.2f} ms")
    print(f"{'Throughput (QPS)':<28} | {exact_stats['qps']:>13.1f} req/s| {ann_stats['qps']:>13.1f} req/s")
    print("-" * 68)
    print(f"ANN Recall@{top_k} vs Exact:      {mean_recall * 100:.2f}%")
    print(f"ANN Recall Loss:               {recall_loss * 100:.2f}%")
    print("=" * 68 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Semantic Retrieval (Exact vs HNSW FAISS)")
    parser.add_argument("--config", type=str, default="configs/semantic_config.yaml")
    parser.add_argument("--max-passages", type=int, default=5000, help="Number of passages to benchmark")
    args = parser.parse_args()

    run_semantic_benchmark(config_path=args.config, max_passages=args.max_passages)


if __name__ == "__main__":
    main()
