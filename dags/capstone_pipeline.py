"""SDAIA capstone pipeline - real Apache Airflow DAG.

    ingest_from_kafka
        -> build_bronze -> build_silver -> build_gold
        -> verify_delta_guarantees
        -> quality_gate_great_expectations
        -> rag_pipeline
        -> pipeline_summary

Each task runs a capstone module in the project's own virtualenv as a
subprocess. That keeps Airflow's dependency set isolated from Spark/Delta/
Chroma (they cannot be resolved together in one environment) while still using
real Airflow operators, real task instances and real dependency semantics.

The quality gate is a normal task that raises on failure. Airflow's default
trigger rule is all_success, so when `quality_gate_great_expectations` fails,
`rag_pipeline` and `pipeline_summary` are never executed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# capstone/ is the parent of dags/
CAPSTONE_ROOT = Path(__file__).resolve().parents[1]

# The interpreter that has pyspark / delta / chromadb / great-expectations.
PIPELINE_PYTHON = os.getenv(
    "CAPSTONE_PYTHON", str(CAPSTONE_ROOT / ".venv311" / "Scripts" / "python.exe")
)


def run_module(module: str, *args: str) -> None:
    """Run a capstone module and fail the Airflow task if it exits non-zero."""
    if not Path(PIPELINE_PYTHON).exists():
        raise FileNotFoundError(
            f"Pipeline interpreter not found: {PIPELINE_PYTHON}. "
            "Set CAPSTONE_PYTHON to the capstone virtualenv's python.exe."
        )

    cmd = [PIPELINE_PYTHON, "-u", "-m", module, *args]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    print(f"[airflow] $ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, cwd=str(CAPSTONE_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    # Decode with utf-8 and replace any unmappable chars, then write safely.
    out = proc.stdout.decode("utf-8", errors="replace")
    # Write to stdout in a way that won't crash on non-ASCII chars.
    try:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    except Exception:
        # Fallback: strip to ASCII if the console codepage can't handle it.
        print(out.encode("ascii", errors="replace").decode("ascii"))

    if proc.returncode != 0:
        # Raising is what makes Airflow mark this task FAILED and skip downstream.
        raise RuntimeError(
            f"Task module {module} exited with code {proc.returncode}. "
            "See the task log above for the failure detail."
        )
    print(f"[airflow] {module} completed successfully")


def summarise() -> None:
    """Print the artefacts produced by a successful run."""
    for rel in (
        "data/bronze/readings", "data/silver/readings", "data/gold/machine_daily_kpi",
        "evidence/quality_gate_result.json", "evidence/rag_evidence.json",
        "evidence/lineage/openlineage_events.jsonl",
    ):
        p = CAPSTONE_ROOT / rel
        print(f"[summary] {'OK  ' if p.exists() else 'MISS'} {rel}")
    print("[summary] capstone pipeline finished end-to-end")


default_args = {"owner": "sdaia-capstone", "retries": 0}

with DAG(
    dag_id="capstone_pipeline",
    description="Kafka -> Delta Bronze/Silver/Gold -> Great Expectations -> RAG",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,          # triggered manually / by `airflow dags test`
    catchup=False,
    max_active_runs=1,
    tags=["sdaia", "capstone", "kafka", "delta", "rag"],
) as dag:

    produce = PythonOperator(
        task_id="produce_seed_data",
        python_callable=run_module,
        op_args=["src.ingestion.producer"],
        doc_md="Publish the demo batch (12 valid + 4 deliberately invalid records) "
               "to the real Kafka topic capstone.readings.",
    )

    ingest = PythonOperator(
        task_id="ingest_from_kafka",
        python_callable=run_module,
        op_args=["src.ingestion.consumer"],
        doc_md="Consume capstone.readings, enforce the Pydantic contract, "
               "route violations to the capstone.dlq dead-letter topic.",
    )

    bronze = PythonOperator(
        task_id="build_bronze",
        python_callable=run_module,
        op_args=["src.lakehouse.bronze"],
        doc_md="Append contract-accepted readings to the Bronze Delta table.",
    )

    silver = PythonOperator(
        task_id="build_silver",
        python_callable=run_module,
        op_args=["src.lakehouse.silver"],
        doc_md="Clean/deduplicate Bronze and UPSERT into Silver with Delta MERGE "
               "keyed on reading_id.",
    )

    gold = PythonOperator(
        task_id="build_gold",
        python_callable=run_module,
        op_args=["src.lakehouse.gold"],
        doc_md="Aggregate Silver into per-machine, per-day KPIs.",
    )

    verify_delta = PythonOperator(
        task_id="verify_delta_guarantees",
        python_callable=run_module,
        op_args=["src.lakehouse.verify_delta"],
        doc_md="Prove the Delta MERGE updates in place and that an incompatible "
               "schema write is refused.",
    )

    quality_gate = PythonOperator(
        task_id="quality_gate_great_expectations",
        python_callable=run_module,
        op_args=["src.quality.gate"],
        doc_md="Validate Gold with Great Expectations. Raises on failure, which "
               "prevents every downstream task from running.",
    )

    rag = PythonOperator(
        task_id="rag_pipeline",
        python_callable=run_module,
        op_args=["src.rag.pipeline"],
        doc_md="Hybrid retrieval (ChromaDB + BM25 + RRF) -> cross-encoder rerank "
               "-> grounded answer with citations.",
    )

    summary = PythonOperator(task_id="pipeline_summary", python_callable=summarise)

    # The quality gate sits between the lakehouse and everything downstream.
    produce >> ingest >> bronze >> silver >> gold >> verify_delta >> quality_gate >> rag >> summary
