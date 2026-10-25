"""Preprocessing pipeline to transform raw documents into processed passages.

Loads raw crawled documents from data/raw/documents.jsonl, splits into passages
using TextChunker, normalizes and tokenizes text using TextPreprocessor,
and saves the structured corpus to data/processed/corpus.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from tqdm import tqdm

from src.preprocess.chunker import TextChunker
from src.preprocess.tokenizer import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocess_pipeline")


def run_pipeline(config_path: str = "configs/preprocess_config.yaml", max_docs: int | None = None) -> Dict[str, Any]:
    """Execute the corpus preprocessing and chunking pipeline."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_path = Path(config["input_corpus"])
    output_path = Path(config["output_corpus_jsonl"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = TextPreprocessor.from_config(config.get("preprocessing", {}))

    chunk_cfg = config.get("chunking", {})
    chunking_enabled = chunk_cfg.get("enabled", True)
    chunker = TextChunker(
        chunk_size_words=chunk_cfg.get("chunk_size_words", 300),
        chunk_overlap_words=chunk_cfg.get("chunk_overlap_words", 50),
        min_chunk_words=chunk_cfg.get("min_chunk_words", 30),
    )

    logger.info("Reading raw corpus from %s...", input_path)
    total_docs = 0
    total_chunks = 0
    total_tokens = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Processing documents"):
            if not line.strip():
                continue
            doc = json.loads(line)
            total_docs += 1

            doc_id = doc["doc_id"]
            title = doc.get("title", "")
            url = doc.get("url", "")
            body_text = doc.get("clean_body_text", "")

            # 1. Chunk document if enabled
            if chunking_enabled:
                chunks = chunker.chunk_document(
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    text=body_text,
                )
            else:
                chunks = [
                    TextChunker(min_chunk_words=1).chunk_document(doc_id, title, url, body_text)[0]
                ] if body_text else []

            # 2. Tokenize and write each chunk
            for chunk in chunks:
                tokens = preprocessor.preprocess(chunk.text)
                if not tokens:
                    continue

                processed_record = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "url": chunk.url,
                    "chunk_index": chunk.chunk_index,
                    "raw_text": chunk.text,
                    "tokens": tokens,
                    "processed_text": " ".join(tokens),
                    "token_count": len(tokens),
                }
                outfile.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                total_chunks += 1
                total_tokens += len(tokens)

            if max_docs is not None and total_docs >= max_docs:
                logger.info("Reached max_docs limit: %d", max_docs)
                break

    stats = {
        "input_documents": total_docs,
        "processed_chunks": total_chunks,
        "total_tokens": total_tokens,
        "mean_tokens_per_chunk": round(total_tokens / total_chunks, 2) if total_chunks else 0,
        "mean_chunks_per_doc": round(total_chunks / total_docs, 2) if total_docs else 0,
    }

    logger.info(
        "Preprocessing complete! Docs: %d | Chunks: %d | Total Tokens: %d | Mean tokens/chunk: %.1f",
        total_docs,
        total_chunks,
        total_tokens,
        stats["mean_tokens_per_chunk"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing and chunking pipeline")
    parser.add_argument("--config", type=str, default="configs/preprocess_config.yaml")
    parser.add_argument("--max-docs", type=int, default=None, help="Process first N docs for testing")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, max_docs=args.max_docs)


if __name__ == "__main__":
    main()
