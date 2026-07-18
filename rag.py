"""Citation-grounded retrieval over SEC filing text.

Pipeline: fetch a filing's narrative text -> section-aware chunking -> hybrid
retrieval (BM25 sparse + TF-IDF dense, optional neural embeddings) -> answer
with inline citations back to the source passages.

Two answer modes:
- extractive (default, free): returns the retrieved passages verbatim as cited
  evidence — it cannot state a number that isn't in the filing.
- generative (optional): Claude writes a grounded answer over the retrieved
  context with [n] citations, enabled only when an Anthropic API key is set.

The dense retriever defaults to TF-IDF (light, always available). Installing
the `embeddings` extra swaps in sentence-transformers; the eval ablation
(eval_rag.py) compares them.
"""
from __future__ import annotations

import bisect
import os
import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 10-K narrative sections we care about; headers double as citation labels.
_ITEM_RE = re.compile(r"\bItem\s+(\d{1,2}[A-Z]?)\.?\s+([A-Z][A-Za-z' ,&/-]{3,60})")

# Canonical 10-K item titles — used for clean, recognizable citation labels
# instead of whatever text the regex happened to capture from the document.
_ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Common Stock",
    "6": "Selected Financial Data", "7": "Management's Discussion & Analysis",
    "7A": "Market Risk Disclosures", "8": "Financial Statements",
    "9": "Changes in Accountants", "9A": "Controls & Procedures",
    "9B": "Other Information", "10": "Directors & Officers",
    "11": "Executive Compensation", "12": "Security Ownership",
    "13": "Related Transactions", "14": "Accountant Fees",
    "15": "Exhibits & Schedules", "16": "Form 10-K Summary",
}
_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_STOPWORDS = frozenset(
    "the a an and or of to in for on with that this these those is are was were be "
    "as by at from its it their our we they would could may might will can what "
    "which who how does do about into during than then".split()
)

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
DEFAULT_MODEL = os.environ.get("SEC_RAG_MODEL", "claude-opus-4-8")


@dataclass
class Chunk:
    text: str
    section: str
    index: int
    source: str  # e.g. "AAPL 10-K (2025-10-31)"
    url: str = ""


@dataclass
class Hit:
    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)  # per-method breakdown


# --- text cleaning & chunking -------------------------------------------------
def clean_document(raw: str) -> str:
    """Strip HTML to readable narrative text."""
    if "<" in raw and ">" in raw:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(" ")
    else:
        text = raw
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _section_markers(text: str) -> list[tuple[int, str]]:
    """(offset, 'Item 1A. Risk Factors') for each section header found."""
    markers: list[tuple[int, str]] = []
    for m in _ITEM_RE.finditer(text):
        num = m.group(1).upper()
        title = _ITEM_TITLES.get(num)  # prefer the canonical, clean title
        if title is None:
            # fall back to the captured text, trimmed to a short title
            title = " ".join(m.group(2).split()[:5]).strip(" ,")
        markers.append((m.start(), f"Item {num} · {title}"))
    markers.sort()
    return markers


def _alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(c.isalpha() or c.isspace() for c in s) / len(s)


def chunk_document(text: str, source: str, url: str = "") -> list[Chunk]:
    """Sliding-window chunks, each tagged with its nearest preceding section."""
    markers = _section_markers(text)
    positions = [p for p, _ in markers]
    labels = [lbl for _, lbl in markers]

    chunks: list[Chunk] = []
    step = CHUNK_CHARS - CHUNK_OVERLAP
    idx = 0
    for start in range(0, len(text), step):
        piece = text[start:start + CHUNK_CHARS].strip()
        if len(piece) < 200 or _alpha_ratio(piece) < 0.6:
            continue  # skip tables / boilerplate / whitespace runs
        if positions:
            i = bisect.bisect_right(positions, start) - 1
            section = labels[i] if i >= 0 else "(front matter)"
        else:
            section = "(document)"
        chunks.append(Chunk(text=piece, section=section, index=idx, source=source, url=url))
        idx += 1
    return chunks


def _stem(t: str) -> str:
    """Light suffix stripping so 'returned'/'returns'/'return' collide (and
    'repurchases'->'repurchase', 'shareholders'->'shareholder'). Consistent on
    query and documents, which is what matters for matching."""
    if t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        t = t[:-1]
    return t


def _tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _WORD_RE.findall(text.lower()) if len(t) > 1]


# --- optional neural embedder (lazy, cached) ----------------------------------
_st_model = None
_st_tried = False


def _neural_embedder():
    """Return a sentence-transformers model if the extra is installed, else None."""
    global _st_model, _st_tried
    if _st_tried:
        return _st_model
    _st_tried = True
    try:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:  # not installed or model download failed
        _st_model = None
    return _st_model


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# --- retriever ----------------------------------------------------------------
class Retriever:
    """Hybrid retrieval over one filing's chunks.

    dense_backend: "tfidf" (default) or "neural" (needs the embeddings extra;
    silently falls back to tfidf if unavailable).
    """

    def __init__(self, chunks: list[Chunk], dense_backend: str = "tfidf"):
        if not chunks:
            raise ValueError("No chunks to index")
        self.chunks = chunks
        texts = [c.text for c in chunks]

        tokenized = [_tokenize(t) for t in texts]
        self._token_sets = [set(toks) for toks in tokenized]
        self._bm25 = BM25Okapi(tokenized)

        self.dense_backend = dense_backend
        self._embeddings = None
        self._vectorizer = None
        if dense_backend == "neural" and _neural_embedder() is not None:
            model = _neural_embedder()
            self._embeddings = model.encode(texts, normalize_embeddings=True,
                                            show_progress_bar=False)
        else:
            self.dense_backend = "tfidf"
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._doc_matrix = self._vectorizer.fit_transform(texts)

    def _dense_scores(self, query: str) -> np.ndarray:
        if self._embeddings is not None:
            q = _neural_embedder().encode([query], normalize_embeddings=True,
                                          show_progress_bar=False)
            return (self._embeddings @ q[0])
        qv = self._vectorizer.transform([query])
        return linear_kernel(qv, self._doc_matrix)[0]

    def search(self, query: str, k: int = 5, method: str = "hybrid") -> list[Hit]:
        bm25 = np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=float)
        dense = np.asarray(self._dense_scores(query), dtype=float)

        if method == "bm25":
            final = bm25
        elif method in ("dense", "tfidf", "neural"):
            final = dense
        else:  # hybrid: normalized BM25 + dense, reranked by query-term coverage
            qc = {t for t in _tokenize(query) if t not in _STOPWORDS}
            if qc:
                cov = np.array([len(qc & ts) / len(qc) for ts in self._token_sets])
            else:
                cov = np.zeros(len(self.chunks))
            final = 0.35 * _minmax(bm25) + 0.35 * _minmax(dense) + 0.30 * cov

        order = np.argsort(final)[::-1][:k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=float(final[i]),
                scores={"bm25": float(bm25[i]), "dense": float(dense[i])},
            )
            for i in order
        ]


def build_retriever_from_document(raw: str, source: str, url: str = "",
                                  dense_backend: str = "tfidf") -> Retriever:
    text = clean_document(raw)
    chunks = chunk_document(text, source=source, url=url)
    return Retriever(chunks, dense_backend=dense_backend)


# --- answering ----------------------------------------------------------------
def _best_sentences(query: str, text: str, n: int = 2) -> str:
    """Pull the sentences from a passage most lexically similar to the query.

    Overlap is scored on content words only (stopwords ignored) so a sentence
    doesn't rank highly just for sharing 'the'/'of' with the question.
    """
    sents = [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 30]
    # Drop a leading fragment left by a mid-sentence chunk boundary (a "sentence"
    # that doesn't start with a capital letter), so answers begin cleanly.
    sents = [s for s in sents if s[:1].isupper()] or sents
    if len(sents) <= n:
        return " ".join(sents).strip() if sents else text.strip()
    q = {t for t in _tokenize(query) if t not in _STOPWORDS}

    def overlap(s: str) -> int:
        return len(q & {t for t in _tokenize(s) if t not in _STOPWORDS})

    ranked = sorted(sents, key=overlap, reverse=True)
    top = {s for s in ranked[:n] if overlap(s) > 0} or {ranked[0]}
    return " ".join(s for s in sents if s in top)  # keep original order


def answer_extractive(query: str, hits: list[Hit]) -> dict:
    """Free/grounded mode: cited source passages, no generation."""
    if not hits:
        return {"answer": "No relevant passages found in this filing.", "citations": []}
    lead = _best_sentences(query, hits[0].chunk.text)
    citations = [
        {"n": i + 1, "section": h.chunk.section, "text": h.chunk.text, "score": h.score}
        for i, h in enumerate(hits)
    ]
    answer = f"{lead} [1]" if lead else hits[0].chunk.text
    return {"answer": answer, "citations": citations, "mode": "extractive"}


ANTHROPIC_SYSTEM = (
    "You answer questions about a company's SEC filing using ONLY the numbered "
    "context passages provided. Cite every claim with the passage number in "
    "brackets, e.g. [2]. If the context does not contain the answer, say so "
    "plainly — do not use outside knowledge or guess. Be concise."
)


def answer_with_claude(query: str, hits: list[Hit], model: str | None = None) -> dict:
    """Optional generative mode. Requires `anthropic` + an API key on the host."""
    import anthropic  # optional dep — only imported when this mode is used

    model = model or DEFAULT_MODEL
    context = "\n\n".join(
        f"[{i + 1}] (from {h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(hits)
    )
    prompt = f"Context passages:\n\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1200,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=ANTHROPIC_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    citations = [
        {"n": i + 1, "section": h.chunk.section, "text": h.chunk.text, "score": h.score}
        for i, h in enumerate(hits)
    ]
    return {"answer": text, "citations": citations, "mode": "claude", "model": model}


def claude_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True
