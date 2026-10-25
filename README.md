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

## Corpus Statistics (Phase 1 Baseline)

All numbers are generated programmatically via `src/crawler/corpus_stats.py` and saved to `reports/results/corpus_stats.json`:

| Metric | Measured Value |
| :--- | :--- |
| **Documents Crawled** | 15 (Initial verified sample batch) |
| **Total Word Count** | 50,826 words |
| **Vocabulary Size** | 6,490 unique tokens |
| **Median Document Length** | 2,143 words |
| **Mean Document Length** | 3,388.40 ± 3,016.76 words |
| **Min Document Length** | 501 words |
| **Max Document Length** | 11,329 words |
| **Duplicate Rate** | 0.00% |
| **Top Frequent Terms** | *learning* (688), *data* (464), *machine* (352), *model* (228), *computer* (186) |

Length distribution plot: `reports/figures/corpus_length_hist.png`

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
