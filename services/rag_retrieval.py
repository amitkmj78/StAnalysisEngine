"""Pure retrieval helpers for the general assistant: chunking, BM25 ranking, and
context packing. No database or model imports, so these can be tested on their own."""

import math
import re
from collections import Counter
from dataclasses import dataclass

TICKER_RE = re.compile(r"\b[A-Z][A-Z.]{0,9}\b")
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "what", "which", "from", "about",
    "does", "how", "why", "when", "have", "has", "had", "its", "their", "they", "them", "any", "all",
    "can", "will", "would", "should", "could", "into", "over", "than", "then", "there", "here", "your",
    "you", "our", "not", "but", "out", "one", "two", "per", "new", "more", "most", "also", "been",
}
CHUNK_CHARS = 900
TOP_K = 6
CONTEXT_CHARS = 5000
CORPUS_LIMIT = 2000
SCORE_UNIVERSE = "All"
TICKER_BONUS = 2.0

@dataclass
class Passage:
    source: str
    text: str
    ticker: str | None = None
    always: bool = False  # included even when it does not rank for the question


def _tokens(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS]


def chunks(text: str) -> list[str]:
    """Paragraph-sized pieces, each at most CHUNK_CHARS (long paragraphs are split on sentences)."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", (text or "").strip()):
        para = para.strip()
        if not para:
            continue
        if len(para) <= CHUNK_CHARS:
            out.append(para)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            if len(current) + len(sentence) > CHUNK_CHARS and current:
                out.append(current.strip())
                current = ""
            current += sentence + " "
        if current.strip():
            out.append(current.strip())
    return out


def rank_passages(question: str, passages: list[Passage], k: int = TOP_K) -> list[Passage]:
    """BM25 over the passages, with a bonus for chunks about a ticker the question names."""
    q_terms = set(_tokens(question))
    question_tickers = {t for t in TICKER_RE.findall(question) if 1 < len(t) <= 10}
    if not passages or (not q_terms and not question_tickers):
        return []
    docs = [_tokens(p.text) for p in passages]
    avg_len = sum(len(d) for d in docs) / len(docs) or 1.0
    df: Counter = Counter()
    for d in docs:
        df.update(set(d) & q_terms)
    n = len(docs)
    k1, b = 1.5, 0.75
    scored = []
    for p, d in zip(passages, docs):
        tf = Counter(d)
        score = 0.0
        for term in q_terms:
            if tf[term] == 0:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(d) / avg_len))
        if p.ticker and p.ticker in question_tickers:
            score += TICKER_BONUS
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:k]]


def build_context(passages: list[Passage], budget: int = CONTEXT_CHARS) -> tuple[str, list[Passage]]:
    lines: list[str] = []
    used: list[Passage] = []
    total = 0
    for i, p in enumerate(passages, 1):
        block = f"[{i}] {p.source}\n{p.text}"
        if total + len(block) > budget:
            break
        lines.append(block)
        used.append(p)
        total += len(block)
    return "\n\n".join(lines), used
