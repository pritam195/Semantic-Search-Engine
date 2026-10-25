# Semantic Search Engine: TF-IDF vs. Embedding-Based Retrieval

A rigorous, reproducible benchmark comparing classical lexical retrieval (TF-IDF, BM25) and dense semantic vector retrieval (Sentence-Transformers, FAISS) on a domain-specific corpus crawled from English Wikipedia under Computer Science, Machine Learning, and Statistics categories.

---

## Phase 1: Polite Crawler & Corpus Building (Completed)

In Phase 1, we established the project architecture, virtual environment (Python 3.11), configuration system, and a robust crawler pipeline.

### What Was Built:
1. **Config-Driven Architecture (`configs/`)**:
   - `configs/crawler_config.yaml`: Seed category URLs, rate-limits, timeouts, and filtering thresholds.
   - `configs/base_config.yaml`: Global directory paths and fixed random seeds (`seed=42`).
2. **Polite, Resumable Wikipedia Crawler (`src/crawler/spider.py`)**:
   - **Robots.txt Compliance**: Strictly respects Wikimedia's `robots.txt` rules using `urllib.robotparser`.
   - **Rate Limiting**: Enforces polite throttling (1–2 req/s) with exponential backoff on HTTP 429/5xx and transient errors.
   - **Descriptive User-Agent**: Configurable contact metadata adhering to Wikimedia etiquette.
   - **URL Normalization**: Canonicalizes schemes, hosts, converts spaces to underscores, strips query params and fragments (`#...`), and filters internal Wikipedia namespaces (`Special:`, `File:`, `Talk:`, etc.).
   - **Deduplication**: Multi-level deduplication via canonical URL sets and SHA-256 body content hashes.
   - **Resumable State Tracker**: Persists the crawl frontier, visited URLs, content hashes, and counters to `data/raw/crawler_state.json`.
   - **Boilerplate Stripper & Text Extractor**: Strips scripts, styles, navigation bars, infoboxes, edit sections, and reference lists, extracting clean titles, structured headings (`h2`, `h3`, `h4`), and body text.
   - **Corpus Filters**: Drops non-English pages and documents with fewer than 100 words.
3. **Corpus Statistics & Visualizer (`src/crawler/corpus_stats.py`)**:
   - Computes document count, word length distribution (min, mean, median, IQR, max), vocabulary size, top terms (excluding English stopwords), and duplicate rates.
   - Exports figures to `reports/figures/corpus_length_hist.png` and metrics to `reports/results/corpus_stats.json`.
4. **Interactive Notebook (`notebooks/01_corpus_stats.ipynb`)**:
   - Clean, reproducible inspection and visual analytics for the corpus.
5. **Unit Test Suite (`tests/test_crawler.py`)**:
   - 16 test cases covering URL normalization, content hashing, HTML boilerplate stripping, headings extraction, word count filtering, foreign language filtering, crawler state serialization, and robots checking.

---

## Corpus Statistics (Phase 1 Final - 5,000 Documents)

All numbers are generated programmatically via `src/crawler/corpus_stats.py` and saved to `reports/results/corpus_stats.json`:

| Metric | Measured Value |
| :--- | :--- |
| **Documents Crawled** | 5,000 |
| **Total Word Count** | 13,185,498 words |
| **Vocabulary Size** | 141,919 unique tokens |
| **Median Document Length** | 1,442 words |
| **Mean Document Length** | 2,637.10 ± 3,379.12 words |
| **Min Document Length** | 101 words |
| **Max Document Length** | 41,893 words |
| **Duplicate Rate** | 7.27% (428 content duplicates filtered) |
| **Top Frequent Terms** | *data* (33,774), *learning* (20,107), *model* (17,815), *information* (16,507), *systems* (15,874), *machine* (15,286), *computer* (14,978) |

Length distribution plot: `reports/figures/corpus_length_hist.png`

---

## Phase 2: Lexical Retrieval & From-Scratch TF-IDF (Completed)

In Phase 2, we built the preprocessing and passage-chunking pipeline, implemented a mathematically exact TF-IDF retriever from scratch in NumPy/SciPy sparse matrices, verified correctness against scikit-learn, integrated BM25, and benchmarked both retrievers across 100 domain queries.

### What Was Built:
1. **Configurable Preprocessing & Chunking (`src/preprocess/`)**:
   - `tokenizer.py`: Lowercase, regex tokenization, punctuation removal, stopword removal, and Porter stemming (`nltk`).
   - `chunker.py`: Sliding-window passage chunker (300 words, 50-word overlap) turning long articles into 54,581 searchable units with preserved parent metadata.
   - `pipeline.py`: Pipeline processing `data/raw/` into `data/processed/corpus.jsonl`.
2. **From-Scratch TF-IDF Retriever (`src/lexical/tfidf_retriever.py`)**:
   - Built purely with NumPy and SciPy CSR sparse matrices (no `TfidfVectorizer` used for retrieval).
   - Inverted index representation with postings.
   - Smoothed IDF: $\text{idf}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1.0$.
   - Sublinear term frequency: $\text{tf}(t, d) = 1 + \ln(\text{count})$ for $\text{count} > 0$.
   - Row-wise $L_2$ normalization and cosine similarity dot product ranking.
3. **BM25 Retriever (`src/lexical/bm25_retriever.py`)**:
   - Parameterized BM25 ($k_1=1.5, b=0.75$) conforming to the uniform `BaseRetriever` interface.
4. **Uniform Retrieval Interface (`src/lexical/base.py`)**:
   - Abstract `BaseRetriever` class enforcing `search(query, k) -> list[(doc_id, score)]` and `search_with_metadata(query, k) -> list[SearchResult]`.
5. **Correctness Verification Suite (`tests/test_lexical.py`)**:
   - Unit tests strictly comparing custom TF-IDF vocabulary, IDF weights, document vectors, query transformations, and cosine ranking scores against `sklearn.feature_extraction.text.TfidfVectorizer` (tolerances $\le 10^{-5}$).
6. **Lexical Benchmark (`src/lexical/benchmark.py`)**:
   - Evaluates indexing speed, index disk footprint, and retrieval latency over 100 queries. Results persisted to `reports/results/lexical_benchmark.json`.

---

## Lexical Retrieval Benchmark (Phase 2 Results)

Evaluated on **54,581 passages** over **100 domain queries** ($k=10$). Programmatic metrics from `reports/results/lexical_benchmark.json`:

| Metric | Custom TF-IDF (NumPy/SciPy) | BM25 Baseline (`rank_bm25`) |
| :--- | :--- | :--- |
| **Indexing Time** | **15.00 s** | **2.21 s** |
| **Index Size on Disk** | **284.05 MB** | **294.18 MB** |
| **Vocabulary Size** | **57,109 features** | **57,109 features** |
| **Non-Zero Matrix Elements** | **5,189,974 (0.167% density)**| N/A |
| **Mean Query Latency** | **5.54 ms** | **85.53 ms** |
| **Median Query Latency** | **5.29 ms** | **84.96 ms** |
| **95th Percentile Latency (p95)** | **6.43 ms** | **112.66 ms** |
| **99th Percentile Latency (p99)** | **6.95 ms** | **118.39 ms** |
| **Throughput (QPS)** | **180.5 req/s** | **11.7 req/s** |

---

## Phase 3: Semantic Retrieval & FAISS Vector Search (Completed)

In Phase 3, we built the dense semantic retrieval pipeline using Sentence-Transformers, dual FAISS indices (Exact Flat and Approximate HNSW), passage-to-document score aggregation, and evaluated recall loss between exact and ANN search.

### What Was Built:
1. **Passage Chunking & Document Aggregation (`src/semantic/retriever.py`)**:
   - Decomposes documents into overlapping passages (200 words, 40-word overlap).
   - Preserves chunk-to-document mappings.
   - Configurable document-level score aggregation strategies: **max-pooling** (default), **mean-pooling**, and **top-k sum**.
2. **Dense Sentence Embedder (`src/semantic/embedder.py`)**:
   - Supports both `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` via ONNX Runtime.
   - Handles the required BGE retrieval query prefix (`"Represent this sentence for searching relevant passages: "`).
   - Generates 384-dimensional $L_2$-normalized dense embeddings for cosine equivalency under inner product.
   - Batch encoding with binary disk caching in `.npy` format.
3. **Dual FAISS Vector Indices (`src/semantic/faiss_index.py`)**:
   - **Exact Search**: `faiss.IndexFlatIP` (exhaustive dot product search).
   - **Approximate Search (ANN)**: `faiss.IndexHNSWFlat` ($M=32, efConstruction=64, efSearch=64$).
4. **Uniform Retrieval Interface**:
   - Conforms strictly to `BaseRetriever`: `search(query, k) -> list[(doc_id, score)]` and `search_with_metadata(query, k) -> list[SearchResult]`.
5. **Semantic Benchmark Suite (`src/semantic/benchmark.py`)**:
   - Profiles embedding time, index build time, memory footprint (RSS), exact vs ANN retrieval latencies over 100 domain queries, and computes empirical **Recall Loss**. Metrics saved to `reports/results/semantic_benchmark.json`.

---

## Semantic Retrieval Benchmark (Phase 3 Results)

Evaluated on **5,000 passages** over **100 domain queries** ($k=10$) with `all-MiniLM-L6-v2` ($D=384$). Programmatic metrics from `reports/results/semantic_benchmark.json`:

| Metric | Exact Dense Search (`IndexFlatIP`) | Approximate Search (`IndexHNSWFlat`) |
| :--- | :--- | :--- |
| **Index Build Time** | **0.02 s** | **0.55 s** |
| **Index Size on Disk** | **7.32 MB** | **8.62 MB** |
| **Embedding Cache (`.npy`)** | **7.32 MB** | **7.32 MB** |
| **Embedding Time (5k passages)** | **351.00 s (14.2 passages/s on CPU)** | **351.00 s** |
| **RAM Footprint (RSS)** | **1,070.94 MB** | **1,070.94 MB** |
| **Mean Query Latency** | **5.95 ms** | **5.00 ms** |
| **Median Query Latency** | **5.00 ms** | **4.41 ms** |
| **95th Percentile Latency (p95)**| **9.41 ms** | **8.19 ms** |
| **99th Percentile Latency (p99)**| **15.15 ms** | **9.47 ms** |
| **Throughput (QPS)** | **168.0 req/s** | **199.9 req/s** |
| **ANN Recall@10 vs Exact** | **100.00% (Baseline)** | **98.10%** |
| **ANN Recall Loss** | **0.00%** | **1.90%** |

---

## How to Run

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/pritam195/Semantic-Search-Engine.git
cd "Semantic Search Engine"

# Create and activate virtual environment (Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Crawler
```bash
# Run crawler with configuration (resumes automatically if interrupted)
python -m src.crawler.spider --config configs/crawler_config.yaml

# Or crawl a quick custom target (e.g., 50 documents)
python -m src.crawler.spider --target-docs 50
```

### 3. Generate Corpus Statistics and Histogram
```bash
python -m src.crawler.corpus_stats
```

### 4. Run Unit Tests
```bash
pytest -v tests/test_crawler.py
```

---

## Sample Crawled Documents

Inspected directly from `data/raw/documents.jsonl`:

1. **Machine learning** (`https://en.wikipedia.org/wiki/Machine_learning`)
   - Length: 11,329 words | Headings: 58 (`History`, `Artificial intelligence`, `Theory`, etc.)
2. **Statistics** (`https://en.wikipedia.org/wiki/Statistics`)
   - Length: 8,034 words | Headings: 23 (`Statistical data`, `Data collection`, `Inference`, etc.)
3. **Computer science** (`https://en.wikipedia.org/wiki/Computer_science`)
   - Length: 5,410 words | Headings: 27 (`History`, `Philosophy`, `Theoretical computer science`, etc.)
4. **Outline of machine learning** (`https://en.wikipedia.org/wiki/Outline_of_machine_learning`)
   - Length: 3,765 words | Headings: 40 (`Paradigms of machine learning`, `Applications`, etc.)
5. **80 Million Tiny Images** (`https://en.wikipedia.org/wiki/80_Million_Tiny_Images`)
   - Length: 501 words | Headings: 2 (`Construction`, `Retirement`)

---

## Repository Structure
```text
├── configs/              # Config-driven YAML files
├── data/
│   ├── raw/              # Raw crawled JSONL and state files
│   ├── processed/        # Cleaned/chunked corpus (Phase 2)
│   └── indexes/          # TF-IDF & FAISS indexes (Phase 3 & 4)
├── src/
│   ├── crawler/          # Spider and corpus stats
│   ├── preprocess/       # Tokenizers and chunkers (Phase 2)
│   ├── lexical/          # TF-IDF and BM25 retrievers (Phase 3)
│   ├── semantic/         # Embedders and FAISS indexes (Phase 4)
│   ├── hybrid/           # Reciprocal Rank Fusion (Phase 5)
│   ├── eval/             # MRR, MAP, NDCG, latency metrics (Phase 6)
│   └── api/              # FastAPI application (Phase 7)
├── notebooks/            # Exploratory and evaluation notebooks
├── reports/
│   ├── figures/          # Plots (histograms, PR curves)
│   └── results/          # Programmatic JSON benchmark outputs
├── tests/                # Pytest unit tests
├── Dockerfile            # Container definition
├── requirements.txt      # Pinned Python dependencies
└── README.md
```
