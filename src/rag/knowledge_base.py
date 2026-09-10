"""Knowledge base for the capstone RAG stage.

Two sources are combined:

1. Static engineering runbook documents (carried over from day03) that explain
   the platform concepts an operator may ask about.
2. Documents generated live from the Gold Delta table, so the RAG stage is
   actually grounded in the pipeline's own output rather than a fixed corpus.
"""
from __future__ import annotations

import re

from src.common import config

# --- Static runbook / concept documents ---------------------------------
RUNBOOK_DOCS: list[dict] = [
    {"id": "kb_kafka_ordering", "source": "runbook/kafka.md", "text": (
        "Apache Kafka stores messages in topics. Each topic is split into "
        "partitions - ordered, append-only logs. Every message inside a "
        "partition gets an offset, which is its permanent address. Ordering is "
        "guaranteed only within a single partition, so records that must stay "
        "ordered have to share a partition key. A consumer group shares "
        "partitions so that each partition is assigned to exactly one member."
    )},
    {"id": "kb_dlq", "source": "runbook/ingestion.md", "text": (
        "The capstone ingestion consumer validates every Kafka record against a "
        "Pydantic contract at the ingestion boundary. Records that violate the "
        "contract are produced to the dead-letter topic capstone.dlq together "
        "with the original payload, the rejection reason and the validation "
        "error. A rejected record never reaches the Bronze layer, so bad data "
        "cannot enter the lakehouse."
    )},
    {"id": "kb_medallion", "source": "runbook/lakehouse.md", "text": (
        "The lakehouse uses the medallion architecture. Bronze stores raw "
        "contract-accepted readings exactly as Kafka delivered them, including "
        "partition and offset. Silver holds cleaned, typed and deduplicated "
        "records, one row per reading_id. Gold contains business aggregates at "
        "one row per machine per day, so Gold is always smaller than Silver."
    )},
    {"id": "kb_merge", "source": "runbook/lakehouse.md", "text": (
        "MERGE INTO applies inserts, updates and deletes atomically in one "
        "operation. The capstone Silver layer is maintained by a Delta MERGE "
        "keyed on the business key reading_id. When a corrected value arrives "
        "for a reading_id that already exists, the existing row is updated in "
        "place rather than duplicated, which is the correct way to implement "
        "change data capture into a Delta table."
    )},
    {"id": "kb_schema_enforcement", "source": "runbook/lakehouse.md", "text": (
        "Delta Lake enforces the table schema on write. Writing a column whose "
        "type conflicts with the existing table fails with "
        "DELTA_FAILED_TO_MERGE_FIELDS and the transaction is refused, so the "
        "rows, the schema and the table version all stay unchanged. Schema "
        "enforcement is what prevents a malformed batch from silently "
        "corrupting a table."
    )},
    {"id": "kb_quality_gate", "source": "runbook/quality.md", "text": (
        "The quality gate validates the Gold aggregate with Great Expectations. "
        "It checks structure, completeness, allowed machine ids, compound "
        "uniqueness of machine_id and reading_date, and value ranges. The "
        "decisive expectation is the operational anomaly-rate ceiling: if any "
        "machine's anomaly_rate exceeds the configured maximum the suite fails, "
        "the gate raises an exception, and Airflow skips every downstream task."
    )},
    {"id": "kb_lineage", "source": "runbook/lineage.md", "text": (
        "Lineage is emitted with the official OpenLineage Python client. Every "
        "pipeline stage emits a START event, then a COMPLETE event on success "
        "or a FAIL event carrying an errorMessage facet when it raises. Events "
        "are written through OpenLineage's FileTransport so they can be kept as "
        "evidence without running an external collector."
    )},
    {"id": "kb_hybrid_search", "source": "runbook/rag.md", "text": (
        "Hybrid search combines vector semantic search with BM25 keyword "
        "search. Reciprocal Rank Fusion merges both ranked lists using "
        "score = sum of 1/(k + rank), where k = 60 is the standard constant. "
        "RRF is parameter-free and consistently outperforms weighted linear "
        "combination because it depends on rank, not on incomparable scores."
    )},
    {"id": "kb_reranking", "source": "runbook/rag.md", "text": (
        "Cross-encoder reranking is the second stage of a two-stage retrieval "
        "pattern. Stage one uses a bi-encoder to retrieve candidates quickly "
        "from independent query and document embeddings. Stage two scores each "
        "query-document pair jointly with a cross-encoder, which is far more "
        "accurate because the model sees the interaction between query and "
        "document tokens."
    )},
    {"id": "kb_airflow", "source": "runbook/orchestration.md", "text": (
        "Airflow orchestrates the capstone pipeline as a single DAG: ingestion, "
        "then Bronze, Silver and Gold, then the quality gate, then the RAG "
        "stage. Dependencies are declared so the RAG task runs only after the "
        "quality gate succeeds. Because Airflow's default trigger rule is "
        "all_success, a failed quality task leaves downstream tasks in the "
        "upstream_failed state and they never execute."
    )},
]


def gold_documents() -> list[dict]:
    """Turn the current Gold KPI rows into retrievable documents.

    Uses the ``deltalake`` Rust reader (no JVM) so this works even when no
    Spark session is active. Returns an empty list if the Gold table does not
    exist yet (e.g. during unit tests that only exercise the static corpus).
    """
    gold_path = config.GOLD_PATH
    if not gold_path.exists():
        return []

    from deltalake import DeltaTable

    dt = DeltaTable(str(gold_path))
    pdf = dt.to_pandas()
    pdf = pdf.sort_values(["machine_id", "reading_date"])

    docs = []
    for r in pdf.itertuples(index=False):
        docs.append({
            "id": f"gold_{r.machine_id}_{r.reading_date}",
            "source": f"delta://gold/machine_daily_kpi/{r.machine_id}/{r.reading_date}",
            "text": (
                f"On {r.reading_date}, machine {r.machine_id} reported "
                f"{r.readings} temperature readings. "
                f"{r.anomaly_count} of them were anomalies, an anomaly rate of "
                f"{r.anomaly_rate}. The average temperature was "
                f"{r.avg_temperature_c} degrees Celsius, ranging from "
                f"{r.min_temperature_c} to {r.max_temperature_c}, "
                f"a thermal spread of {r.temp_spread_c} degrees."
            ),
        })
    return docs


def build_corpus(include_gold: bool = True) -> list[dict]:
    docs = list(RUNBOOK_DOCS)
    if include_gold:
        docs.extend(gold_documents())
    return docs


def chunk_documents(docs: list[dict], sentences_per_chunk: int = 2) -> list[dict]:
    """Split documents into overlapping chunks (adjacent chunks share a sentence)."""
    chunks: list[dict] = []
    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc["text"].strip())
        step = max(1, sentences_per_chunk - 1)
        for i in range(0, len(sentences), step):
            text = " ".join(sentences[i: i + sentences_per_chunk]).strip()
            if not text:
                continue
            chunks.append({
                "id": f"{doc['id']}#c{i:02d}",
                "doc_id": doc["id"],
                "source": doc["source"],
                "text": text,
            })
            if i + sentences_per_chunk >= len(sentences):
                break
    return chunks
