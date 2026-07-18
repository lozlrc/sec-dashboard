"""RAG unit tests on a synthetic corpus — no network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag

# A clean, non-repeated 10-K excerpt padded to span multiple chunks.
DOC = (
    "Item 1. Business\n"
    "The company designs and sells consumer electronics and services worldwide. "
    "Competition in the smartphone and personal computer markets is intense, and "
    "the company competes on product features, design, price, and quality. " + ("Filler context sentence about products and services. " * 20) + "\n\n"
    "Item 1A. Risk Factors\n"
    "The company's operations and financial results are subject to various risks "
    "and uncertainties that could adversely affect its business, including supply "
    "chain disruptions and macroeconomic conditions. " + ("Additional risk disclosure sentence describing uncertainty. " * 20) + "\n\n"
    "Item 7. Management's Discussion and Analysis\n"
    "Net sales increased during the fiscal year driven by higher iPhone revenue and "
    "growth in the Services segment. The company returned capital to shareholders "
    "through dividends and share repurchases. " + ("Management commentary sentence about margins and revenue. " * 20)
)

# Distinct, well-separated passages for deterministic ranking tests.
CHUNKS = [
    rag.Chunk(text="Competition in the company's markets is intense; it competes with "
              "other manufacturers on product features, design, and price.",
              section="Item 1. Business", index=0, source="TEST 10-K"),
    rag.Chunk(text="The business is subject to risks and uncertainties, including supply "
              "chain disruption and foreign exchange volatility, that could adversely "
              "affect financial results.",
              section="Item 1A. Risk Factors", index=1, source="TEST 10-K"),
    rag.Chunk(text="Net sales and revenue increased year over year, and gross margin "
              "expanded due to a favorable product mix.",
              section="Item 7. MD&A", index=2, source="TEST 10-K"),
    rag.Chunk(text="The company returned capital to shareholders through quarterly "
              "dividends and an ongoing share repurchase program.",
              section="Item 7. MD&A", index=3, source="TEST 10-K"),
]


def test_chunking_labels_sections():
    text = rag.clean_document(DOC)
    chunks = rag.chunk_document(text, source="TEST 10-K")
    assert len(chunks) >= 3, "expected several chunks"
    sections = {c.section for c in chunks}
    assert any("Item 1A" in s for s in sections), sections
    assert any("Item 7" in s for s in sections), sections


def test_retrieval_ranks_relevant_passage_first():
    r = rag.Retriever(CHUNKS)
    cases = [
        ("What does the filing say about competition?", "competit"),
        ("What are the main risk factors and uncertainties?", "risk"),
        ("How did net sales and revenue change?", "net sales"),
        ("How does the company return capital to shareholders?", "dividend"),
    ]
    for query, keyword in cases:
        hits = r.search(query, k=4, method="hybrid")
        assert keyword in hits[0].chunk.text.lower(), \
            f"top hit for {query!r} missing {keyword!r}: {hits[0].chunk.text!r}"


def test_methods_all_return_ranked_hits():
    r = rag.Retriever(CHUNKS)
    for method in ("bm25", "tfidf", "hybrid"):
        hits = r.search("risk factors and uncertainties", k=3, method=method)
        assert len(hits) == 3
        assert hits[0].score >= hits[-1].score  # descending
        assert "risk" in hits[0].chunk.text.lower()


def test_extractive_answer_is_grounded_and_cited():
    r = rag.Retriever(CHUNKS)
    q = "What are the company's risk factors?"
    hits = r.search(q, k=3)
    ans = rag.answer_extractive(q, hits)
    assert ans["mode"] == "extractive"
    assert ans["citations"], "expected citations"
    assert "[1]" in ans["answer"]
    # answer text is drawn verbatim from the top retrieved passage (grounded)
    body = ans["answer"].replace("[1]", "").strip()
    assert body in hits[0].chunk.text


def test_empty_hits_handled():
    ans = rag.answer_extractive("anything", [])
    assert ans["citations"] == []
