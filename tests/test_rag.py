"""Tests for the RAG pipeline: retrieval, reranking, answer generation, citations."""
from __future__ import annotations

import json

import pytest

from src.rag.knowledge_base import build_corpus, chunk_documents
from src.rag.retrieval import (
    build_vector_index,
    dense_search,
    keyword_search,
    reciprocal_rank_fusion,
    rerank,
    hybrid_retrieve,
)
from src.rag.generate import generate_answer


@pytest.fixture(scope="module")
def corpus_chunks():
    docs = build_corpus()
    chunks = chunk_documents(docs)
    return docs, chunks


@pytest.fixture(scope="module")
def vector_index(corpus_chunks):
    _, chunks = corpus_chunks
    return build_vector_index(chunks)


class TestCorpus:
    def test_corpus_has_documents(self, corpus_chunks):
        docs, chunks = corpus_chunks
        assert len(docs) > 0
        assert len(chunks) > 0

    def test_chunks_have_ids_and_sources(self, corpus_chunks):
        _, chunks = corpus_chunks
        for c in chunks:
            assert "id" in c
            assert "source" in c
            assert "text" in c


class TestDenseRetrieval:
    def test_dense_returns_relevant_results(self, vector_index):
        results = dense_search(vector_index, "How does the DLQ prevent bad records?", top_k=4)
        assert len(results) > 0
        assert any("dlq" in r["text"].lower() or "dead" in r["text"].lower() for r in results)


class TestBM25Retrieval:
    def test_bm25_returns_relevant_results(self, corpus_chunks):
        _, chunks = corpus_chunks
        results = keyword_search(chunks, "MERGE reading_id upsert", top_k=4)
        assert len(results) > 0
        assert any("merge" in r["text"].lower() for r in results)


class TestHybridRetrieval:
    def test_rrf_fuses_dense_and_bm25(self, vector_index, corpus_chunks):
        _, chunks = corpus_chunks
        query = "What happens when Great Expectations fails?"
        dense = dense_search(vector_index, query, top_k=8)
        bm25 = keyword_search(chunks, query, top_k=8)
        fused = reciprocal_rank_fusion(dense, bm25, top_k=8)
        assert len(fused) > 0
        for r in fused:
            assert "rrf_score" in r
            assert "id" in r


class TestReranking:
    def test_crossencoder_reranks(self, vector_index, corpus_chunks):
        _, chunks = corpus_chunks
        query = "Why is Gold smaller than Silver?"
        dense = dense_search(vector_index, query, top_k=8)
        bm25 = keyword_search(chunks, query, top_k=8)
        fused = reciprocal_rank_fusion(dense, bm25, top_k=8)
        reranked = rerank(query, fused, top_k=4)
        assert len(reranked) <= 4
        assert len(reranked) > 0
        assert "gold" in reranked[0]["text"].lower() or "silver" in reranked[0]["text"].lower()


class TestAnswerGeneration:
    def test_answer_is_grounded_with_citations(self, vector_index, corpus_chunks):
        _, chunks = corpus_chunks
        query = "How does the DLQ prevent malformed records from entering the lakehouse?"
        dense = dense_search(vector_index, query, top_k=8)
        bm25 = keyword_search(chunks, query, top_k=8)
        fused = reciprocal_rank_fusion(dense, bm25, top_k=8)
        reranked = rerank(query, fused, top_k=4)
        result = generate_answer(query, reranked)
        answer = result["answer"]
        citations = result["citations"]
        assert len(answer) > 20, "answer should be substantive"
        assert "[Source" in answer, "answer should contain inline citations"
        assert len(citations) > 0, "citations list should not be empty"
        assert result["grounded"] is True
        for c in citations:
            assert c["marker"].startswith("[Source")


class TestHybridRetrieveFunnel:
    def test_full_funnel_returns_all_stages(self, vector_index, corpus_chunks):
        _, chunks = corpus_chunks
        result = hybrid_retrieve(vector_index, chunks, "What is the anomaly rate?")
        assert "dense" in result
        assert "keyword" in result
        assert "fused" in result
        assert "reranked" in result
        assert len(result["dense"]) > 0
        assert len(result["keyword"]) > 0
        assert len(result["fused"]) > 0
        assert len(result["reranked"]) > 0
