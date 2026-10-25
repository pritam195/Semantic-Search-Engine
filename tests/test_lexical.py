"""Unit tests verifying from-scratch TF-IDF against scikit-learn and BM25 retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.lexical.bm25_retriever import BM25Retriever
from src.lexical.tfidf_retriever import CustomTFIDFRetriever
from src.preprocess.tokenizer import TextPreprocessor


@pytest.fixture
def sample_corpus() -> List[dict]:
    return [
        {
            "doc_id": "doc1",
            "title": "Machine Learning",
            "clean_body_text": "machine learning builds mathematical models for prediction using training data and optimization",
        },
        {
            "doc_id": "doc2",
            "title": "Deep Neural Networks",
            "clean_body_text": "neural networks and deep learning architectures with backpropagation and gradient descent",
        },
        {
            "doc_id": "doc3",
            "title": "Statistical Inference",
            "clean_body_text": "statistical inference estimation hypothesis testing and probability distribution models",
        },
        {
            "doc_id": "doc4",
            "title": "Computer Vision",
            "clean_body_text": "computer vision image processing feature extraction convolutional neural networks models",
        },
        {
            "doc_id": "doc5",
            "title": "Support Vector Machine",
            "clean_body_text": "support vector machine kernel trick margin maximization classification algorithm",
        },
    ]


class TestCustomTfidfVsSklearn:
    """Verify correctness: Compare from-scratch NumPy/SciPy TF-IDF against sklearn."""

    @pytest.mark.parametrize("sublinear_tf", [False, True])
    def test_correctness_against_sklearn(self, sample_corpus: List[dict], sublinear_tf: bool) -> None:
        """Verify vocabulary, IDF values, document vectors, and query cosine scores match sklearn."""
        # Simple whitespace tokenizer for strict 1:1 comparison
        prep = TextPreprocessor(lowercase=True, remove_punctuation=False, stopwords=None, stemmer="none")
        raw_texts = [d["clean_body_text"] for d in sample_corpus]

        # 1. Custom Implementation
        custom = CustomTFIDFRetriever(
            smooth_idf=True,
            sublinear_tf=sublinear_tf,
            norm="l2",
            min_df=1,
            max_df=1.0,
            preprocessor=prep,
        )
        custom.index(sample_corpus)

        # 2. Sklearn Reference Implementation
        sklearn_vec = TfidfVectorizer(
            smooth_idf=True,
            sublinear_tf=sublinear_tf,
            norm="l2",
            min_df=1,
            max_df=1.0,
            token_pattern=r"(?u)\b\w+\b",
        )
        sklearn_matrix = sklearn_vec.fit_transform(raw_texts)

        # A. Vocabulary alignment
        assert custom.vocabulary == sklearn_vec.vocabulary_

        # B. IDF vector match
        np.testing.assert_allclose(
            custom.idf_,
            sklearn_vec.idf_,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Custom IDF weights do not match sklearn TfidfVectorizer.idf_",
        )

        # C. Document vectors match
        custom_dense = custom.doc_matrix.toarray()
        sklearn_dense = sklearn_matrix.toarray()
        np.testing.assert_allclose(
            custom_dense,
            sklearn_dense,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Custom TF-IDF matrix does not match sklearn matrix",
        )

        # D. Query Vectorization & Cosine Similarity Ranking Match
        queries = [
            "neural networks deep learning",
            "statistical hypothesis testing",
            "support vector machine classification",
            "machine learning models",
        ]

        for query in queries:
            custom_results = custom.search(query, k=5)

            # Compute reference scores with sklearn
            sklearn_q = sklearn_vec.transform([query])
            sklearn_scores = (sklearn_matrix @ sklearn_q.T).toarray().ravel()

            # Verify scores for returned docs
            doc_id_to_idx = {d["doc_id"]: i for i, d in enumerate(sample_corpus)}

            for doc_id, custom_score in custom_results:
                idx = doc_id_to_idx[doc_id]
                expected_score = float(sklearn_scores[idx])
                assert pytest.approx(expected_score, abs=1e-5) == custom_score

            # Verify top rank matches
            if len(custom_results) > 0:
                best_doc_id, _ = custom_results[0]
                expected_best_idx = int(np.argmax(sklearn_scores))
                expected_best_doc = sample_corpus[expected_best_idx]["doc_id"]
                assert best_doc_id == expected_best_doc

    def test_save_and_load(self, sample_corpus: List[dict], tmp_path: Path) -> None:
        index_dir = tmp_path / "tfidf_index"
        retriever = CustomTFIDFRetriever(sublinear_tf=True, smooth_idf=True)
        retriever.index(sample_corpus)

        query = "neural networks gradient descent"
        orig_results = retriever.search(query, k=3)

        retriever.save(index_dir)
        assert (index_dir / "tfidf_matrix.npz").exists()
        assert (index_dir / "metadata.json").exists()

        loaded_retriever = CustomTFIDFRetriever()
        loaded_retriever.load(index_dir)

        loaded_results = loaded_retriever.search(query, k=3)
        assert orig_results == loaded_results


class TestBM25Retriever:
    """Test BM25 retrieval functionality and ranking."""

    def test_indexing_and_search(self, sample_corpus: List[dict]) -> None:
        bm25 = BM25Retriever(k1=1.5, b=0.75)
        bm25.index(sample_corpus)

        results = bm25.search("neural networks deep architectures", k=3)
        assert len(results) > 0
        # doc2 ("Deep Neural Networks") should be ranked top
        assert results[0][0] == "doc2"
        assert results[0][1] > 0.0

    def test_save_and_load(self, sample_corpus: List[dict], tmp_path: Path) -> None:
        index_dir = tmp_path / "bm25_index"
        bm25 = BM25Retriever()
        bm25.index(sample_corpus)

        query = "statistical inference estimation"
        orig_results = bm25.search(query, k=3)

        bm25.save(index_dir)
        assert (index_dir / "bm25_model.pkl").exists()

        loaded_bm25 = BM25Retriever()
        loaded_bm25.load(index_dir)

        loaded_results = loaded_bm25.search(query, k=3)
        assert orig_results == loaded_results
