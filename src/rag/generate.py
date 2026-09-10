"""Answer generation grounded in the reranked retrieval context, with citations.

Two backends, selected by CAPSTONE_LLM_BACKEND:

  "extractive" (default)
      A local, dependency-free grounded generator. It selects the sentences from
      the retrieved context that best answer the question (cross-encoder ranking
      of candidate sentences against the query) and composes the answer strictly
      from those sentences, appending the [Source N] marker of the chunk each
      sentence came from. Because every sentence is copied from retrieved
      context, the answer cannot hallucinate.

  "openai"
      Sends the same context to an OpenAI-compatible chat completion endpoint.
      Credentials come from the environment only (OPENAI_API_KEY, optional
      OPENAI_BASE_URL, CAPSTONE_LLM_MODEL). Nothing is hardcoded. If the key is
      absent this backend raises instead of pretending to have called an LLM.

Both backends return the same structure, so the pipeline and tests are identical.
"""
from __future__ import annotations

import os
import re

from src.common import config

CITATION_RE = re.compile(r"\[Source (\d+)\]")


def build_context_block(reranked: list[dict]) -> tuple[str, list[dict]]:
    """Render the numbered context block and the citation map."""
    lines, citations = [], []
    for i, chunk in enumerate(reranked, start=1):
        lines.append(f"[Source {i}] {chunk['text']}")
        citations.append({
            "marker": f"[Source {i}]",
            "chunk_id": chunk["id"],
            "doc_id": chunk["doc_id"],
            "source": chunk["source"],
        })
    return "\n\n".join(lines), citations


def build_prompt(query: str, context_block: str) -> str:
    return (
        "You are a data platform engineer answering questions about the SDAIA "
        "capstone pipeline. Answer ONLY from the context below. Cite every "
        "claim with the matching [Source N] marker. If the context does not "
        "contain the answer, reply exactly: "
        "'The retrieved context does not contain this information.'\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )


# --------------------------------------------------------------------------
# Backend 1: local grounded extractive generation
# --------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def generate_extractive(query: str, reranked: list[dict], max_sentences: int = 3) -> str:
    """Compose an answer from the highest-scoring context sentences.

    Sentences are ranked against the query with the same cross-encoder used for
    reranking, so selection is a real relevance model rather than a heuristic.
    """
    from src.rag.retrieval import rerank as _rerank_chunks

    # Treat every sentence of the retrieved context as a candidate, keeping the
    # [Source N] marker of the chunk it came from. Chunks overlap by design, so
    # the same sentence can appear more than once - keep only its first
    # occurrence (the highest-ranked chunk) to avoid a repetitive answer.
    candidates: list[dict] = []
    seen: set[str] = set()
    for idx, chunk in enumerate(reranked, start=1):
        for sent in _split_sentences(chunk["text"]):
            fingerprint = re.sub(r"\W+", " ", sent.lower()).strip()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append({
                "id": f"{chunk['id']}::s{len(candidates)}",
                "text": sent,
                "doc_id": chunk["doc_id"],
                "source": chunk["source"],
                "marker": f"[Source {idx}]",
            })

    if not candidates:
        return "The retrieved context does not contain this information."

    ranked = _rerank_chunks(query, candidates, top_k=max_sentences)

    parts = []
    for cand in ranked:
        sentence = cand["text"].rstrip()
        marker = cand["marker"]
        # Cite the source for each borrowed sentence.
        if sentence.endswith("."):
            sentence = f"{sentence[:-1]} {marker}."
        else:
            sentence = f"{sentence} {marker}"
        parts.append(sentence)

    return " ".join(parts)


# --------------------------------------------------------------------------
# Backend 2: OpenAI-compatible chat completion
# --------------------------------------------------------------------------
def generate_openai(query: str, context_block: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CAPSTONE_LLM_BACKEND=openai but OPENAI_API_KEY is not set. "
            "Export the key or use the default extractive backend."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is not installed. "
            "pip install openai, or use CAPSTONE_LLM_BACKEND=extractive."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
    response = client.chat.completions.create(
        model=os.getenv("CAPSTONE_LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": build_prompt(query, context_block)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------------------------------
def generate_answer(query: str, reranked: list[dict]) -> dict:
    """Generate a grounded, cited answer from the reranked context."""
    context_block, citations = build_context_block(reranked)
    backend = os.getenv("CAPSTONE_LLM_BACKEND", "extractive").lower()

    if backend == "openai":
        answer = generate_openai(query, context_block)
    elif backend == "extractive":
        answer = generate_extractive(query, reranked)
    else:
        raise ValueError(f"unknown CAPSTONE_LLM_BACKEND {backend!r}")

    # Keep only the citations the answer actually used, and verify each marker
    # maps back to a chunk that was really retrieved.
    used_indexes = {int(n) for n in CITATION_RE.findall(answer)}
    used_citations = [c for i, c in enumerate(citations, start=1) if i in used_indexes]

    return {
        "query": query,
        "backend": backend,
        "answer": answer,
        "citations": used_citations,
        "all_context_citations": citations,
        "context_chunk_ids": [c["id"] for c in reranked],
        "grounded": bool(used_citations),
    }
