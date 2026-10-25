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

## Phase 4: Rigorous Retrieval Evaluation (Completed)

In Phase 4, we built an end-to-end evaluation benchmark across 90 stratified queries, generated pooled candidate judgments (2,384 relevance pairs), implemented IR metrics from scratch, ran paired bootstrap hypothesis tests with 95% confidence intervals, and performed deep error analysis.

### What Was Built:
1. **Stratified Query Benchmark (`data/eval/queries.jsonl`)**:
   - 90 queries categorized into 3 distinct representation types:
     - **30 Keyword queries**: Exact technical terminology, acronyms, and named entities.
     - **30 Natural Language Questions**: Conversational and instructional information requests.
     - **30 Paraphrase / Vocabulary Mismatch queries**: Descriptions and analogies with zero-to-low token overlap with canonical documents.
2. **Relevance Pooling & Labeling Tools (`src/eval/labeling.py`)**:
   - Union pooling of top-10 candidates from all retrievers (TF-IDF, BM25, MiniLM, BGE) resulting in 2,384 candidate document-query pairs in `data/eval/qrels.jsonl`.
   - Graded relevance scale: `0` (not relevant), `1` (relevant), `2` (highly relevant).
   - Interactive CLI tool (`python -m src.eval.labeling --cli`) and Cohen's Kappa agreement calculation.
3. **From-Scratch IR Metrics Suite (`src/eval/metrics.py`)**:
   - Exact mathematical implementations of Precision@k, Recall@k, MRR, MAP, and nDCG@k with unit test verification against hand-calculated toy matrices.
4. **Statistical Hypothesis Testing (`src/eval/stats.py`)**:
   - Paired bootstrap significance testing ($B=1000$ iterations) and empirical 95% confidence intervals.
5. **Visualization Engine (`src/eval/plots.py`)**:
   - `reports/figures/metrics_by_query_type.png`: Grouped bar charts across query categories.
   - `reports/figures/latency_vs_ndcg.png`: Latency vs accuracy Pareto tradeoff frontier.
   - `reports/figures/recall_at_k_curves.png`: Multi-model Recall@k curves ($k=1 \dots 10$).

---

## Rigorous Evaluation Benchmark (Phase 4 Results)

Evaluated across **90 benchmark queries** over the **5,000 document collection** ($k=10$). Programmatic metrics from `reports/results/evaluation_results.json`:

| Retriever | Overall nDCG@10 | 95% Confidence Interval | MAP | MRR | Precision@10 | Recall@10 | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF** | 0.4363 | [0.384, 0.491] | 0.2824 | 0.7284 | 0.4156 | 0.3925 | 10.55 ms |
| **BM25** | 0.4707 | [0.408, 0.528] | 0.2410 | 0.7181 | 0.3500 | 0.3213 | 86.48 ms |
| **MiniLM** | 0.5590 | [0.506, 0.612] | 0.3897 | 0.8058 | 0.5289 | 0.5437 | 10.86 ms |
| **BGE** | **0.6122** | **[0.555, 0.665]** | **0.4053** | **0.8175** | **0.5244** | **0.5358** | 16.48 ms |

---

## Performance Stratification by Query Type (nDCG@10)

| Retriever | Keyword Queries (N=30) | Natural Language Questions (N=30) | Paraphrase / Vocab Mismatch (N=30) |
| :--- | :--- | :--- | :--- |
| **TF-IDF** | 0.5431 | 0.5353 | 0.2304 |
| **BM25** | 0.6224 | 0.5320 | 0.2578 |
| **MiniLM** | 0.7058 | 0.5704 | 0.4007 |
| **BGE** | **0.7908** | **0.6032** | **0.4426** |

---

## Statistical Significance Testing (Paired Bootstrap, B=1000)

| Comparison | Mean Difference ($\Delta$) | Bootstrap p-value | Significance |
| :--- | :--- | :--- | :--- |
| **BM25 vs. TF-IDF** | +0.0344 | $p = 0.1330$ | Not Significant ($p \ge 0.05$) |
| **MiniLM vs. BM25** | **+0.0883** | **$p = 0.0060$** | **Statistically Significant ($p < 0.01$)** |
| **BGE vs. BM25** | **+0.1414** | **$p = 0.0000$** | **Statistically Significant ($p < 0.001$)** |
| **BGE vs. MiniLM** | +0.0532 | $p = 0.0550$ | Marginally Significant ($p \approx 0.05$) |

---

## Error Analysis: Why Each Approach Wins

### 1. Where Lexical Retrieval (BM25 / TF-IDF) Wins:
- **Rare Named Entities and Acronyms**: For queries like `"gradient boosting xgboost residuals"` (q22: BM25 0.7802 vs MiniLM 0.4539), the exact token `xgboost` carries massive inverse document frequency ($\text{IDF}$), allowing BM25 to isolate the exact page instantly.
- **Specific Mathematical Jargon**: For `"how does boosting sequentially fit new models to the pseudo-residuals of past models?"` (q52: BM25 0.9639 vs MiniLM 0.0829), the term `pseudo-residuals` is uniquely discriminative in lexical space. Embedding models compress the phrase into a generic ensemble representation, retrieving broad machine learning articles instead of the precise gradient boosting formulation.
- **Parameter Nuances**: In `"gradient descent learning rate step size"` (q05: BM25 0.8121 vs MiniLM 0.4228), lexical matching guarantees all 5 technical keywords co-occur.

### 2. Where Semantic Embeddings (MiniLM / BGE) Win:
- **Zero-Overlap Vocabulary Mismatch**:
  - *"drawing a decision boundary that keeps the widest possible corridor between categories"* (q61: BM25 0.0000 vs MiniLM 0.6740). BM25 fails completely ($nDCG=0$) because words like "corridor" and "categories" never appear in the formal definition of Support Vector Machines. MiniLM maps the geometric intuition directly to SVM margin maximization!
  - *"allowing every token in a sentence to look at and score its relationship with every other token"* (q76: BM25 0.0000 vs MiniLM 0.6309). BM25 scores 0; MiniLM matches the attention mechanism of Transformers.
  - *"crowdsourcing decisions by polling a large collection of randomized branching flowcharts"* (q67: BM25 0.0000 vs MiniLM 0.5714). BM25 scores 0; MiniLM identifies Random Forest decision trees.
- **Conceptual Invariance**:
  - *"measuring angle between two word frequency vectors regardless of text length"* (q88: BM25 0.0000 vs MiniLM 0.5900). Semantic embeddings identify Cosine Similarity without the term appearing.

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
