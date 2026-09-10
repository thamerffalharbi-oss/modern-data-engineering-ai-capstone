"""Tests for the Airflow DAG: structure, dependencies, downstream stop."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def dag():
    """Import the DAG module without initialising the full Airflow DB."""
    dag_path = Path(__file__).resolve().parents[1] / "dags" / "capstone_pipeline.py"
    spec = importlib.util.spec_from_file_location("capstone_pipeline", dag_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dag


class TestDAGStructure:
    def test_dag_id(self, dag):
        assert dag.dag_id == "capstone_pipeline"

    def test_dag_has_all_tasks(self, dag):
        task_ids = {t.task_id for t in dag.tasks}
        expected = {
            "produce_seed_data", "ingest_from_kafka", "build_bronze",
            "build_silver", "build_gold", "verify_delta_guarantees",
            "quality_gate_great_expectations", "rag_pipeline", "pipeline_summary",
        }
        assert task_ids == expected

    def test_task_count(self, dag):
        assert len(dag.tasks) == 9


class TestDAGDependencies:
    def test_linear_dependency_chain(self, dag):
        """Tasks must be chained in the correct pipeline order."""
        deps = {t.task_id: set(t.downstream_task_ids) for t in dag.tasks}
        assert deps["produce_seed_data"] == {"ingest_from_kafka"}
        assert deps["ingest_from_kafka"] == {"build_bronze"}
        assert deps["build_bronze"] == {"build_silver"}
        assert deps["build_silver"] == {"build_gold"}
        assert deps["build_gold"] == {"verify_delta_guarantees"}
        assert deps["verify_delta_guarantees"] == {"quality_gate_great_expectations"}
        assert deps["quality_gate_great_expectations"] == {"rag_pipeline"}
        assert deps["rag_pipeline"] == {"pipeline_summary"}
        assert deps["pipeline_summary"] == set()

    def test_quality_gate_blocks_rag(self, dag):
        """The quality gate must sit between the lakehouse and RAG."""
        gate = dag.get_task("quality_gate_great_expectations")
        assert "rag_pipeline" in gate.downstream_task_ids
        rag = dag.get_task("rag_pipeline")
        assert "quality_gate_great_expectations" in rag.upstream_task_ids

    def test_downstream_stop_on_failure(self, dag):
        """If quality_gate fails, rag_pipeline and pipeline_summary cannot run.

        With Airflow's default trigger rule (all_success), a failed upstream
        task means all downstream tasks are marked upstream_failed / skipped.
        """
        gate = dag.get_task("quality_gate_great_expectations")
        rag = dag.get_task("rag_pipeline")
        summary = dag.get_task("pipeline_summary")

        # The trigger rule for all tasks should be all_success (the default).
        assert gate.trigger_rule == "all_success"
        assert rag.trigger_rule == "all_success"
        assert summary.trigger_rule == "all_success"
