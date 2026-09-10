"""Hybrid retrieval: ChromaDB dense search + BM25 keyword search + RRF + rerank.

Carried over from the working day03 implementation and packaged so each stage
returns its intermediate ranking. Exposing every stage is deliberate: the
evidence must show dense results, keyword results, fused results and reranked
results separately.
"""
from __future__ import annotations

import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.common import config

# Reuse the model cache day03 populated so retrieval works offline.
os.environ.setdefault("HF_HOME", str(config.HF_CACHE))

_reranker: CrossEncoder | None = None


def build_vector_index(chunks: list[dict], collection_name: str = "capstone_kb"):
    """Embed chunks into a real ChromaDB collection."""
    ef = SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL)
    client = chromadb.Client()
    # Start clean so repeated runs do not accumulate duplicate ids.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection did not exist yet
    collection = client.create_collection(collection_name, embedding_function=ef)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"doc_id": c["doc_id"], "source": c["source"]} for c in chunks],
    )
    return collection


def dense_search(collection, query: str, top_k: int = 8) -> list[dict]:
    raw = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for rank, (cid, doc, meta, dist) in enumerate(
        zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]), start=1
    ):
        hits.append({
            "id": cid, "text": doc, "source": meta["source"],
            "doc_id": meta["doc_id"], "rank": rank, "distance": round(float(dist), 4),
        })
    return hits


def keyword_search(chunks: list[dict], query: str, top_k: int = 8) -> list[dict]:
    """BM25 keyword search over the same chunks."""
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "id": chunks[idx]["id"], "text": chunks[idx]["text"],
            "source": chunks[idx]["source"], "doc_id": chunks[idx]["doc_id"],
            "rank": rank, "bm25_score": round(float(score), 4),
        }
        for rank, (idx, score) in enumerate(ranked, start=1)
    ]


def reciprocal_rank_fusion(
    dense_hits: list[dict], keyword_hits: list[dict], k: int = 60, top_k: int = 8
) -> list[dict]:
    """Fuse two ranked lists with RRF: score = sum 1/(k + rank)."""
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    contributions: dict[str, list[str]] = {}

    for source_name, hits in (("dense", dense_hits), ("bm25", keyword_hits)):
        for hit in hits:
            cid = hit["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + hit["rank"])
            payload.setdefault(cid, hit)
            contributions.setdefault(cid, []).append(f"{source_name}@{hit['rank']}")

    fused = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
    return [
        {
            "id": cid, "text": payload[cid]["text"], "source": payload[cid]["source"],
            "doc_id": payload[cid]["doc_id"], "rank": rank,
            "rrf_score": round(scores[cid], 6),
            "retrieved_by": contributions[cid],
        }
        for rank, cid in enumerate(fused, start=1)
    ]


def rerank(query: str, candidates: list[dict], top_k: int = 4) -> list[dict]:
    """Rerank candidates with a real cross-encoder."""
    global _reranker
    if not candidates:
        return []
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANK_MODEL)
    scores = _reranker.predict([(query, c["text"]) for c in candidates])
    ordered = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)[:top_k]
    return [
        {**cand, "rank": rank, "cross_encoder_score": round(float(score), 4)}
        for rank, (score, cand) in enumerate(ordered, start=1)
    ]


def hybrid_retrieve(collection, chunks: list[dict], query: str,
                    top_k_retrieve: int = 8, top_k_final: int = 4) -> dict:
    """Run the full retrieval funnel and return every intermediate stage."""
    dense = dense_search(collection, query, top_k_retrieve)
    keyword = keyword_search(chunks, query, top_k_retrieve)
    fused = reciprocal_rank_fusion(dense, keyword, top_k=top_k_retrieve)
    reranked = rerank(query, fused, top_k_final)
    return {"query": query, "dense": dense, "keyword": keyword,
            "fused": fused, "reranked": reranked}
