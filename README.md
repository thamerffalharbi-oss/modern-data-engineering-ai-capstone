# Modern Data Engineering for AI Systems — Capstone Project

**SDAIA Academy**  
**Delivered via Learning Space**

[![SDAIA Academy](https://img.shields.io/badge/SDAIA-Academy-blue)](https://github.com/SDAIAAcademy)

A unified, end-to-end data pipeline that uses **real** technologies — not mocks,
not simulations, not Python queues pretending to be Kafka.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
  - [Kafka](#kafka)
  - [Delta Lake — Bronze / Silver / Gold](#delta-lake--bronze--silver--gold)
  - [Delta MERGE](#delta-merge)
  - [RAG](#rag)
  - [Hybrid Search](#hybrid-search)
  - [Reranking](#reranking)
  - [Great Expectations](#great-expectations)
  - [Apache Airflow](#apache-airflow)
  - [OpenLineage](#openlineage)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [How to Run](#how-to-run)
- [How to Run Tests](#how-to-run-tests)
- [Success Scenario](#success-scenario)
- [Failure Scenario](#failure-scenario)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)
- [Evidence](#evidence)
- [Limitations](#limitations)
- [SDAIA Attribution](#sdaia-attribution)

---

## Project Overview

This capstone implements a production-grade data engineering pipeline for
industrial IoT sensor data (machine temperature readings). It covers the full
lifecycle: real-time ingestion from Kafka, medallion architecture in Delta
Lake, automated quality gates, AI-powered retrieval (RAG), lineage tracking,
and Airflow orchestration.

Every technology is real and verified:
- **Kafka**: `confluent-kafka` with a live broker
- **Delta Lake**: `pyspark` + `delta-spark` with MERGE and schema enforcement
- **RAG**: `ChromaDB` + `BM25` + `RRF` + `CrossEncoder` reranking
- **Quality**: `Great Expectations` with 14 expectations
- **Lineage**: `openlineage-python` with START/COMPLETE/FAIL events
- **Orchestration**: `Apache Airflow` DAG with 9 tasks

## Problem Statement

Manufacturing facilities generate continuous streams of sensor readings
(temperature, vibration, pressure). The challenge is to build a reliable
data platform that:

1. **Ingests** high-frequency sensor data from Kafka with a strict data contract
2. **Stores** data in a Delta Lakehouse with Bronze/Silver/Gold layers
3. **Ensures quality** through automated Great Expectations validation
4. **Provides answers** about the pipeline and its data via a RAG system
5. **Tracks lineage** so every data transformation is auditable
6. **Orchestrates** the entire flow as a single Airflow DAG

Invalid records (unknown machines, impossible temperatures, missing fields)
must be rejected at the ingestion boundary and routed to a dead-letter queue
(DLQ) — they must never enter the lakehouse.

---

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌─────────────────────────────┐    ┌───────────┐    ┌──────┐
│  Kafka   │───▶│  Ingestion   │───▶│  Delta Lakehouse            │───▶│  GE Gate  │───▶│ RAG  │
│ producer │    │  Pydantic    │    │  Bronze → Silver → Gold     │    │  pass/fail│    │      │
│ + consumer│   │  + DLQ       │    │  + MERGE + schema enforce   │    │           │    │      │
└──────────┘    └──────────────┘    └─────────────────────────────┘    └───────────┘    └──────┘
                         │                       │                          │              │
                         ▼                       ▼                          ▼              ▼
                   accepted.jsonl          Delta tables              evidence JSON    evidence JSON
                   rejected.jsonl          (parquet + _delta_log)    + lineage       + citations
```

Airflow orchestrates the entire flow as a single DAG with explicit task
dependencies. The quality gate sits between the lakehouse and RAG; if it
fails, every downstream task is skipped.

## Pipeline

### Kafka

- **Producer** (`src/ingestion/producer.py`): publishes a deterministic batch
  of 16 records (12 valid + 4 deliberately malformed) to `capstone.readings`
  using `confluent-kafka`.
- **Consumer** (`src/ingestion/consumer.py`): consumes from `capstone.readings`,
  validates each record against the Pydantic contract, and routes:
  - Valid records → `data/ingest/accepted.jsonl` (input to Bronze)
  - Invalid records → real Kafka DLQ topic `capstone.dlq` + `rejected.jsonl`
- **Contract** (`src/ingestion/contract.py`): `MachineReading` Pydantic model
  enforces machine_id whitelist, temperature range [-50, 200], required fields,
  and timestamp format.

### Delta Lake — Bronze / Silver / Gold

| Layer  | Grain                | Operation         | Key columns                                  |
|--------|----------------------|-------------------|----------------------------------------------|
| Bronze | One row per Kafka msg| Append (immutable)| reading_id, machine_id, temperature_c, kafka_offset |
| Silver | One row per reading_id| Delta MERGE (upsert)| + reading_date, status, is_anomaly           |
| Gold   | One row per machine/day| Aggregate        | readings, anomaly_count, avg/max/min temp, anomaly_rate, temp_spread |

- **Bronze** (`src/lakehouse/bronze.py`): raw accepted readings, appended as-is
  with Kafka provenance (partition, offset) and ingestion timestamp.
- **Silver** (`src/lakehouse/silver.py`): cleaned, typed, deduplicated. Maintained
  by real `DeltaTable.merge()` keyed on `reading_id`.
- **Gold** (`src/lakehouse/gold.py`): genuine aggregate — 12 Silver rows → 3 Gold
  rows (one per machine per day).

### Delta MERGE

Silver uses a real Delta `MERGE INTO` operation keyed on the business key
`reading_id`. When a corrected value arrives for an existing reading:

1. The existing row is **updated in place** (temperature, status, anomaly flag)
2. **No duplicate row** is created
3. Delta history records `numTargetRowsMatchedUpdated: 1`

Verified: `CNC_MILL_01-01` updated from 67.0°C (NORMAL) → 97.5°C (ANOMALY)
without row count change.

### RAG

The RAG pipeline (`src/rag/`) answers natural-language questions about the
data platform and its Gold KPIs:

1. **Knowledge base** (`knowledge_base.py`): 9 static runbook documents + live
   Gold-derived documents (one per machine-day KPI row).
2. **Chunking**: documents split into overlapping 2-sentence chunks (29 total).
3. **Embeddings**: `all-MiniLM-L6-v2` via `SentenceTransformerEmbeddingFunction`.
4. **Vector store**: real `ChromaDB` collection.
5. **Answer generation**: extractive backend (local, no API key) or optional
   OpenAI-compatible backend. Answers are grounded in retrieved context with
   inline `[Source N]` citations.

### Hybrid Search

Two independent retrieval strategies are run in parallel:

- **Dense retrieval** (`dense_search`): ChromaDB vector similarity search
  (semantic matching via sentence embeddings).
- **Keyword retrieval** (`keyword_search`): BM25Okapi over tokenized chunk text
  (exact term matching).

Results are fused using **Reciprocal Rank Fusion (RRF)**:
`score = sum of 1/(k + rank)` where `k = 60`.

RRF is parameter-free and consistently outperforms weighted linear combination
because it depends on rank position, not on incomparable raw scores.

### Reranking

After RRF fusion, the top candidates are reranked with a real **CrossEncoder**
(`cross-encoder/ms-marco-MiniLM-L-6-v2`). The cross-encoder scores each
query-document pair jointly (seeing the full interaction between query and
document tokens), which is far more accurate than bi-encoder similarity alone.

### Great Expectations

The quality gate (`src/quality/gate.py`) uses the real Great Expectations
library with an `EphemeralDataContext` and a named expectation suite
(`capstone_gold_kpi_suite`). 14 expectations cover:

- **Structure**: column set, row count bounds
- **Completeness**: no nulls in key columns
- **Validity**: machine_id in allowed set, readings type check
- **Uniqueness**: compound (machine_id, reading_date) uniqueness
- **Ranges**: readings, anomaly_count, avg_temperature bounds
- **Pair logic**: max_temperature ≥ min_temperature
- **The gate**: anomaly_rate ≤ 0.60 (operational ceiling)

On failure, `QualityGateFailed` is raised → Airflow marks the task as FAILED
→ all downstream tasks are skipped.

### Apache Airflow

The DAG (`dags/capstone_pipeline.py`) has 9 tasks with explicit linear
dependencies:

```
produce_seed_data → ingest_from_kafka → build_bronze → build_silver → build_gold
→ verify_delta_guarantees → quality_gate_great_expectations → rag_pipeline → pipeline_summary
```

- Each task runs a capstone module as a subprocess (isolates Spark/Delta/Chroma
  deps from Airflow's own dependency tree).
- Default trigger rule `all_success`: a failed task leaves downstream tasks
  in `upstream_failed` state.
- Verified via `airflow dags test` (sequential execution without scheduler).

### OpenLineage

Every pipeline stage emits OpenLineage-compatible events using the official
`openlineage-python` client:

- **START**: when a stage begins
- **COMPLETE**: when a stage succeeds
- **FAIL**: when a stage raises, with an `errorMessage` facet containing the
  exception type and message

Events are written through OpenLineage's `FileTransport` (officially supported)
to `evidence/lineage/openlineage_events.jsonl`. To ship to a real backend
(Marquez), set `CAPSTONE_LINEAGE_TRANSPORT=http` and `OPENLINEAGE_URL`.

---

## Installation

### Prerequisites

| Component   | Version used      | Notes                                      |
|-------------|-------------------|--------------------------------------------|
| Python      | 3.11.9            | 3.13 is not compatible with PySpark 3.5   |
| Java        | OpenJDK 17        | Required by Spark                          |
| PySpark     | 3.5.6             | With delta-spark 3.3.2                     |
| Kafka       | 3.x (native)      | Or Docker Compose if available             |
| Airflow     | 2.10.4            | Separate venv (conflicts with Spark deps)  |

### Step 1 — Create the pipeline virtual environment

```powershell
cd capstone
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2 — Start Kafka

**Option A: Docker Compose** (if Docker is available):

```powershell
docker compose -f infra/docker-compose.yml up -d
```

**Option B: Native Kafka** (Windows without Docker):

```powershell
# Download Kafka from https://kafka.apache.org/downloads
# Extract to C:\kfk\kafka

# Start the KRaft broker
.\infra\kafka-start.ps1

# Create the topics
.\infra\kafka-topics-create.ps1
```

### Step 3 — Set up Airflow (separate venv)

```powershell
python -m venv .venv_airflow
.\.venv_airflow\Scripts\Activate.ps1
pip install apache-airflow pytest
```

---

## Environment Variables

All configuration is env-overridable. See `.env.example` for the full list.

| Variable                    | Default                              | Purpose                        |
|-----------------------------|--------------------------------------|--------------------------------|
| `KAFKA_BOOTSTRAP_SERVERS`   | `localhost:9092`                     | Kafka broker address           |
| `KAFKA_TOPIC_READINGS`      | `capstone.readings`                  | Source topic                   |
| `KAFKA_TOPIC_DLQ`           | `capstone.dlq`                       | Dead-letter queue topic        |
| `CAPSTONE_MAX_ANOMALY_RATE` | `0.60`                               | GE anomaly-rate ceiling        |
| `CAPSTONE_ANOMALY_THRESHOLD_C` | `85.0`                            | Temperature anomaly threshold  |
| `CAPSTONE_SCENARIO`         | `healthy`                            | Producer scenario (healthy/overheat) |
| `CAPSTONE_LLM_BACKEND`      | `extractive`                         | RAG answer backend             |
| `CAPSTONE_PYTHON`           | `.venv311/Scripts/python.exe`        | Airflow subprocess interpreter |

---

## How to Run

### Manual pipeline execution

```powershell
.\.venv311\Scripts\Activate.ps1

# 1. Publish demo batch to Kafka (12 valid + 4 invalid)
python -m src.ingestion.producer

# 2. Consume, validate, route to DLQ
python -m src.ingestion.consumer

# 3. Build Delta lakehouse
python -m src.lakehouse.bronze
python -m src.lakehouse.silver
python -m src.lakehouse.gold

# 4. Verify MERGE + schema enforcement
python -m src.lakehouse.verify_delta

# 5. Quality gate
python -m src.quality.gate

# 6. RAG pipeline
python -m src.rag.pipeline
```

### Airflow orchestration

```powershell
$env:AIRFLOW_HOME = "$PWD\.airflow_home"
$env:AIRFLOW__CORE__DAGS_FOLDER = "$PWD\dags"
$env:AIRFLOW__CORE__LOAD_EXAMPLES = "False"
$env:CAPSTONE_PYTHON = "$PWD\.venv311\Scripts\python.exe"

python -m airflow db migrate
python -m airflow dags test capstone_pipeline 2024-01-01
```

## How to Run Tests

```powershell
# Non-Airflow tests (31 tests: ingestion, lakehouse, RAG, quality, lineage)
.\.venv311\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_airflow.py

# Airflow tests (6 tests: DAG structure + dependencies)
.\.venv_airflow\Scripts\python.exe -m pytest tests/test_airflow.py -v
```

**Total: 37 tests, all passing.**

---

## Success Scenario

1. Producer publishes 16 records to Kafka (12 valid + 4 invalid)
2. Consumer accepts 12, rejects 4 to DLQ with rejection reasons
3. Bronze: 12 rows appended
4. Silver: 12 rows (cleaned, typed, deduplicated)
5. Gold: 3 rows (per-machine per-day aggregate)
6. Delta MERGE verified: updates in place, no duplicates
7. Schema enforcement verified: incompatible write rejected
8. Great Expectations: 14/14 expectations pass (anomaly_rate max = 0.25)
9. RAG: 4 questions answered with citations
10. OpenLineage: START + COMPLETE for every stage
11. Airflow: DagRun state = **success**

## Failure Scenario

1. Set `CAPSTONE_SCENARIO=overheat` → anomaly_rate = 0.75
2. Pipeline runs through Gold normally
3. Great Expectations: 13/14 pass, `anomaly_rate` expectation **FAILS**
4. `QualityGateFailed` exception raised
5. Airflow marks `quality_gate_great_expectations` as **FAILED**
6. `rag_pipeline` and `pipeline_summary` are **skipped** (upstream_failed)
7. OpenLineage: **FAIL** event emitted with `errorMessage` facet
8. Airflow: DagRun state = **failed**

## Expected Outputs

### Quality gate (success)
```json
{
  "success": true,
  "expectations_evaluated": 14,
  "expectations_passed": 14,
  "observed_max_anomaly_rate": 0.25
}
```

### Quality gate (failure)
```json
{
  "success": false,
  "expectations_evaluated": 14,
  "expectations_passed": 13,
  "observed_max_anomaly_rate": 0.75,
  "failed_expectations": [{"expectation": "expect_column_values_to_be_between", "column": "anomaly_rate"}]
}
```

### RAG answer (with citations)
```
On 2024-03-01, machine CNC_MILL_01 reported 4 temperature readings [Source 1].
1 of them were anomalies, an anomaly rate of 0.25 [Source 1].
```

### OpenLineage events
```json
{"eventType":"START","job":{"namespace":"sdaia-capstone","name":"capstone.ingestion"},...}
{"eventType":"COMPLETE","job":{"namespace":"sdaia-capstone","name":"capstone.ingestion"},...}
{"eventType":"FAIL","job":{"namespace":"sdaia-capstone","name":"capstone.quality"},
 "run":{"facets":{"errorMessage":{"message":"QualityGateFailed: ..."}}},...}
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `JAVA_GATEWAY_EXITED` | JAVA_HOME not set | Run through `run.ps1` or set `JAVA_HOME` to JDK 17 |
| `Failed to acquire idempotence PID` | Kafka broker starting | Transient; producer retries automatically |
| ChromaDB telemetry warning | Chroma version mismatch | Non-fatal; retrieval still works |
| Spark `ShutdownHookManager` IOException | Windows JAR lock | Benign cleanup warning; exit code 0 |
| Airflow `RuntimeWarning: POSIX only` | Windows platform | Use `airflow dags test`; full scheduler needs WSL2 |
| `No active Spark session` in subprocess | Spark stopped between calls | Quality gate uses `deltalake` reader (no JVM) |
| `PATH_NOT_FOUND` for Gold table | Pipeline not run yet | Run bronze → silver → gold first |

---

## Evidence

Run artifacts are stored under `capstone/evidence/`:

| File                              | What it proves                              |
|-----------------------------------|---------------------------------------------|
| `quality_gate_result.json`        | GE pass: 14/14 expectations, 3 Gold rows    |
| `quality_gate_failure_result.json`| GE fail: 13/14, anomaly_rate 0.75 > 0.60    |
| `rag_evidence.json`               | Answers with citations, retrieval metrics   |
| `lineage/openlineage_events.jsonl`| START/COMPLETE events for every stage       |
| `lineage/openlineage_failure_events.jsonl` | Includes FAIL event with errorMessage facet |
| `airflow_dag_test.log`            | Full successful DAG run log                 |
| `airflow_dag_failure_test.log`    | Full failure DAG run (quality gate stops)   |
| `pytest_full_results.log`         | 31 non-Airflow tests passing                |
| `pytest_airflow_results.log`      | 6 Airflow tests passing                     |
| `final_verification_summary.json` | Complete verification summary               |

---

## Limitations

1. **Airflow on Windows**: Airflow officially supports POSIX systems only.
   It runs on Windows via `airflow dags test` (sequential execution without
   scheduler). Full scheduler execution requires WSL2 or Linux containers.
   The DAG structure, dependencies, and downstream stop behaviour are fully
   verified through `airflow dags test`.

2. **OpenLineage backend**: Events are emitted through OpenLineage's
   FileTransport (an officially supported transport). No external collector
   (Marquez) is running. To ship to a real backend, set
   `CAPSTONE_LINEAGE_TRANSPORT=http` and `OPENLINEAGE_URL`.

3. **LLM answer generation**: The default backend is a local extractive
   generator that composes answers strictly from retrieved context sentences
   (no hallucination possible). An OpenAI-compatible backend is available via
   `CAPSTONE_LLM_BACKEND=openai` + `OPENAI_API_KEY`.

4. **Docker/WSL**: Not available in this environment. Native Kafka is used
   instead. A `docker-compose.yml` template can be added for reproducibility
   on systems that have Docker.

---

## Training Program Attribution

| Field | Value |
|-------|-------|
| **Programme** | Modern Data Engineering for AI Systems |
| **Institution** | SDAIA Academy |
| **Delivery** | Delivered via Learning Space |
| **Trainer** | Mohammed Albeladi |
| **Cohort/Session Dates** | [TO BE CONFIRMED] |
| **Academy GitHub** | [https://github.com/SDAIAAcademy](https://github.com/SDAIAAcademy) |

This capstone was completed as the final project for the Modern Data Engineering
for AI Systems programme at SDAIA Academy, delivered via Learning Space and
taught by trainer Mohammed Albeladi.

The original lab exercises (day01–day04) are preserved in their respective
folders; the `capstone/` directory is the unified submission.
