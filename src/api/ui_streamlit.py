"""Interactive Streamlit Demonstration Application for Semantic Search Engine.

Features:
1. Search box with customizable top-k slider.
2. Side-by-side columns comparing TF-IDF vs Dense Semantic (MiniLM) retrieval.
3. Overlap highlighting: Visually labels documents that appear in both lists vs unique hits.
4. Explainability:
   - For TF-IDF: Exact per-query term contribution breakdown (q_t * d_t) with percentages.
   - For Semantic: Best-matching passage chunk excerpt.
5. "Phase 4 Failure Examples" interactive gallery preloaded with error analysis queries.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Set

import streamlit as st

# Add workspace root to sys.path so src imports work cleanly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.api.service import PHASE4_FAILURE_EXAMPLES, SearchEngineService

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Neural vs. Lexical Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics, badges, and cards
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .result-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 10px rgba(0,0,0,0.07);
    }
    .badge-shared {
        background-color: #DCFCE7;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #BBF7D0;
    }
    .badge-tfidf-only {
        background-color: #F3E8FF;
        color: #6B21A8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #E9D5FF;
    }
    .badge-semantic-only {
        background-color: #FFEDD5;
        color: #9A3412;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #FED7AA;
    }
    .score-badge {
        background-color: #F1F5F9;
        color: #334155;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .snippet-text {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.5;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading indices and neural models into memory...")
def get_initialized_service() -> SearchEngineService:
    """Load indices once and cache in Streamlit session."""
    service = SearchEngineService.get_instance()
    service.initialize()
    return service


service = get_initialized_service()

# --- Sidebar: Phase 4 Failure Examples Gallery ---
st.sidebar.title("🔬 Failure Cases Gallery")
st.sidebar.markdown(
    "Preloaded queries from **Phase 4 Error Analysis** demonstrating when lexical vs. semantic search succeeds or fails."
)

selected_example = None
for ex in PHASE4_FAILURE_EXAMPLES:
    if st.sidebar.button(f"📌 {ex['category']}: {ex['target_topic']}", key=ex["id"]):
        selected_example = ex

# --- Main App Header ---
st.markdown('<div class="main-header">⚡ Neural vs. Lexical Search Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Evaluating Classical Inverted Index (TF-IDF/BM25) against Dense Vectors (MiniLM/FAISS) on 5,000 Wikipedia CS/ML/Stats Articles</div>',
    unsafe_allow_html=True,
)

# Active Failure Callout Banner if an example was clicked
if selected_example:
    st.info(
        f"**Selected Failure Case**: *{selected_example['target_topic']}* ({selected_example['category']})\n\n"
        f"• **Lexical Failure Mode**: {selected_example['why_lexical_fails']}\n\n"
        f"• **Semantic Win Mode**: {selected_example['why_semantic_wins']}"
    )

# --- Search Bar & Controls ---
col_search, col_k = st.columns([5, 1])

default_query = selected_example["query"] if selected_example else "support vector machine margin classification"
with col_search:
    query_input = st.text_input(
        "Enter natural language query or technical terms:",
        value=default_query,
        key="search_query_input",
    )

with col_k:
    top_k = st.slider("Top K", min_value=3, max_value=20, value=10)

if not query_input.strip():
    st.warning("Please enter a search query above.")
    st.stop()

# --- Execute Comparison Search ---
with st.spinner("Retrieving and comparing across index spaces..."):
    comp_data = service.compare(query_input.strip(), k=top_k)

tfidf_results = comp_data["methods"]["tfidf"]["results"]
semantic_results = comp_data["methods"]["semantic"]["results"]
tfidf_latency = comp_data["methods"]["tfidf"]["latency_ms"]
semantic_latency = comp_data["methods"]["semantic"]["latency_ms"]

tfidf_doc_ids = [r["doc_id"] for r in tfidf_results]
semantic_doc_ids = [r["doc_id"] for r in semantic_results]

shared_ids = set(tfidf_doc_ids) & set(semantic_doc_ids)
tfidf_only_ids = set(tfidf_doc_ids) - shared_ids
semantic_only_ids = set(semantic_doc_ids) - shared_ids

overlap_pct = (len(shared_ids) / max(1, top_k)) * 100.0

# --- High-Level Comparison Metrics Banner ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{len(shared_ids)} / {top_k}</div><div class="metric-lbl">Overlap ({overlap_pct:.0f}%)</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{len(tfidf_only_ids)}</div><div class="metric-lbl">TF-IDF Only</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{len(semantic_only_ids)}</div><div class="metric-lbl">Semantic Only</div></div>',
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{tfidf_latency:.1f} ms</div><div class="metric-lbl">TF-IDF Latency</div></div>',
        unsafe_allow_html=True,
    )
with m5:
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{semantic_latency:.1f} ms</div><div class="metric-lbl">Semantic Latency</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# Helper maps for cross-rank lookups
semantic_rank_map = {r["doc_id"]: r["rank"] for r in semantic_results}
tfidf_rank_map = {r["doc_id"]: r["rank"] for r in tfidf_results}

# --- Side-by-Side Results Display ---
col_left, col_right = st.columns(2)

# LEFT COLUMN: TF-IDF
with col_left:
    st.subheader(f"📊 TF-IDF Lexical Retrieval ({len(tfidf_results)} hits)")
    st.caption("Exact term matching, smoothed IDF, and sublinear TF weighting via sparse CSR matrix.")

    if not tfidf_results:
        st.warning("No lexical matches found for this query.")

    for item in tfidf_results:
        d_id = item["doc_id"]
        is_shared = d_id in shared_ids

        badge_html = (
            f'<span class="badge-shared">🟢 Also in Semantic (# {semantic_rank_map[d_id]})</span>'
            if is_shared
            else '<span class="badge-tfidf-only">🟣 TF-IDF Only</span>'
        )

        with st.container():
            st.markdown(
                f"""
                <div class="result-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong>#{item['rank']}</strong> <a href="{item['url']}" target="_blank" style="text-decoration:none; font-weight:600; color:#2563EB;">{item['title']}</a>
                        </div>
                        <div>
                            {badge_html} &nbsp; <span class="score-badge">Score: {item['score']:.4f}</span>
                        </div>
                    </div>
                    <div class="snippet-text">{item['snippet']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Explainability Dropdown for TF-IDF
            top_terms = item.get("top_terms", [])
            if top_terms:
                with st.expander(f"🔍 Term Contributions for #{item['rank']} ({item['title']})"):
                    st.write("**Top Contributing Query Terms ($q_t \\times d_t$):**")
                    for t in top_terms:
                        col_term, col_bar = st.columns([2, 5])
                        with col_term:
                            st.write(f"`{t['term']}`: **{t['weight']:.4f}** ({t['percentage']}%)")
                        with col_bar:
                            st.progress(min(1.0, max(0.0, t["percentage"] / 100.0)))

# RIGHT COLUMN: Semantic
with col_right:
    st.subheader(f"🧠 Semantic Dense Retrieval ({len(semantic_results)} hits)")
    st.caption("Sentence-Transformers (MiniLM-L6-v2) 384-d dense vectors with FAISS Inner Product.")

    if not semantic_results:
        st.warning("No semantic vector matches found.")

    for item in semantic_results:
        d_id = item["doc_id"]
        is_shared = d_id in shared_ids

        badge_html = (
            f'<span class="badge-shared">🟢 Also in TF-IDF (# {tfidf_rank_map[d_id]})</span>'
            if is_shared
            else '<span class="badge-semantic-only">🟠 Semantic Only</span>'
        )

        with st.container():
            st.markdown(
                f"""
                <div class="result-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong>#{item['rank']}</strong> <a href="{item['url']}" target="_blank" style="text-decoration:none; font-weight:600; color:#2563EB;">{item['title']}</a>
                        </div>
                        <div>
                            {badge_html} &nbsp; <span class="score-badge">Cosine: {item['score']:.4f}</span>
                        </div>
                    </div>
                    <div class="snippet-text">{item['snippet']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Explainability Dropdown for Semantic
            with st.expander(f"🎯 Best Matching Chunk for #{item['rank']} ({item['title']})"):
                st.markdown(f"**Passage Identifier**: `{item.get('best_chunk_id', 'N/A')}`")
                st.write(item.get("best_chunk", item["snippet"]))
