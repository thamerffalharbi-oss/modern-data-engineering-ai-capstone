"""Capstone RAG stage: hybrid retrieval -> rerank -> grounded cited answer.

Runs the demo questions, records every retrieval stage plus the final answer and
citations to evidence/rag_evidence.json, and verifies that each answer is
grounded in chunks that were actually retrieved.
"""
from __future__ import annotations

import json

from src.common import config
from src.lineage.emitter import lineage_stage
from src.rag.generate import generate_answer
from src.rag.knowledge_base import build_corpus, chunk_documents
from src.rag.retrieval import build_vector_index, hybrid_retrieve

DEMO_QUESTIONS = [
    "How does the pipeline stop malformed Kafka records from reaching the lakehouse?",
    "Why is the Gold layer smaller than the Silver layer, and what does MERGE do?",
    "What happens to downstream tasks when the Great Expectations quality gate fails?",
    "What was the anomaly rate for CNC_MILL_01 and what temperatures did it report?",
]


def _summarise(hits: list[dict], score_key: str | None = None) -> list[dict]:
    out = []
    for h in hits:
        item = {"rank": h["rank"], "chunk_id": h["id"], "source": h["source"]}
        if score_key and score_key in h:
            item[score_key] = h[score_key]
        if "retrieved_by" in h:
            item["retrieved_by"] = h["retrieved_by"]
        out.append(item)
    return out


def run_rag(questions: list[str] | None = None, include_gold: bool = True) -> dict:
    questions = questions or DEMO_QUESTIONS

    docs = build_corpus(include_gold=include_gold)
    chunks = chunk_documents(docs)
    print(f"[rag] corpus: {len(docs)} documents -> {len(chunks)} chunks")

    collection = build_vector_index(chunks)
    print(f"[rag] ChromaDB collection holds {collection.count()} chunks")

    results = []
    for q in questions:
        print(f"\n{'=' * 70}\n[rag] QUERY: {q}\n{'=' * 70}")
        stages = hybrid_retrieve(collection, chunks, q)

        print(f"  dense (ChromaDB):  {len(stages['dense'])} hits, "
              f"top={stages['dense'][0]['id'] if stages['dense'] else '-'}")
        print(f"  keyword (BM25):    {len(stages['keyword'])} hits, "
              f"top={stages['keyword'][0]['id'] if stages['keyword'] else '-'}")
        print(f"  RRF fused:         {len(stages['fused'])} hits, "
              f"top={stages['fused'][0]['id'] if stages['fused'] else '-'}")
        print(f"  cross-encoder:     {len(stages['reranked'])} hits, "
              f"top={stages['reranked'][0]['id'] if stages['reranked'] else '-'}")

        generated = generate_answer(q, stages["reranked"])
        print(f"\n  ANSWER: {generated['answer']}")
        print(f"  CITATIONS: {[c['marker'] + ' -> ' + c['chunk_id'] for c in generated['citations']]}")

        # Every citation must point at a chunk that retrieval actually returned.
        retrieved_ids = {c["id"] for c in stages["reranked"]}
        for cite in generated["citations"]:
            if cite["chunk_id"] not in retrieved_ids:
                raise AssertionError(
                    f"Citation {cite['marker']} references {cite['chunk_id']}, "
                    "which was not in the retrieved context."
                )
        if not generated["grounded"]:
            raise AssertionError(f"Answer for {q!r} contains no citation.")

        results.append({
            "query": q,
            "dense_results": _summarise(stages["dense"], "distance"),
            "bm25_results": _summarise(stages["keyword"], "bm25_score"),
            "rrf_fused_results": _summarise(stages["fused"], "rrf_score"),
            "reranked_results": _summarise(stages["reranked"], "cross_encoder_score"),
            "retrieved_context": [
                {"marker": f"[Source {i}]", "chunk_id": c["id"],
                 "source": c["source"], "text": c["text"]}
                for i, c in enumerate(stages["reranked"], start=1)
            ],
            "final_answer": generated["answer"],
            "citations": generated["citations"],
            "backend": generated["backend"],
            "grounded": generated["grounded"],
        })

    config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EVIDENCE_DIR / "rag_evidence.json"
    out.write_text(json.dumps({
        "documents": len(docs), "chunks": len(chunks), "questions": len(results),
        "results": results,
    }, indent=2), encoding="utf-8")

    print(f"\n[rag] {len(results)} questions answered, all grounded with citations")
    print(f"[rag] evidence -> {out}")
    return {"questions": len(results), "evidence": str(out)}


def main() -> None:
    with lineage_stage("capstone.rag"):
        run_rag()


if __name__ == "__main__":
    main()
