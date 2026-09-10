"""Tests for the Delta Lakehouse: Bronze/Silver/Gold, MERGE, schema enforcement.

These tests build the medallion layers from a synthetic accepted.jsonl so they
do not depend on a running Kafka broker. They use real PySpark + delta-spark.

A single Spark session is shared across all tests in the module. The build
functions (build_bronze, build_silver, build_gold) normally stop Spark in a
finally block; here we monkey-patch get_spark to return the shared session and
suppress stop so the JVM stays alive for verification reads.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.common import config


# ---------------------------------------------------------------------------
# Helpers to build a small accepted.jsonl in the test data dir.
# ---------------------------------------------------------------------------

def _write_accepted(records: list[dict]) -> None:
    config.ensure_dirs()
    config.INGEST_ACCEPTED.parent.mkdir(parents=True, exist_ok=True)
    config.INGEST_ACCEPTED.write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )


def _sample_records(n_valid: int = 6, scenario: str = "healthy") -> list[dict]:
    """Build records in the same shape the consumer writes to accepted.jsonl."""
    from src.ingestion.producer import demo_batch
    from src.ingestion.contract import MachineReading
    batch = demo_batch(scenario)
    accepted = []
    for r in batch:
        try:
            reading = MachineReading(**r)
        except Exception:
            continue  # skip contract violations (they'd go to the DLQ)
        rec = reading.model_dump(mode="json")
        rec["is_anomaly"] = reading.is_anomaly
        rec["kafka_partition"] = 0
        rec["kafka_offset"] = len(accepted)
        accepted.append(rec)
    return accepted[:n_valid]


# ---------------------------------------------------------------------------
# Module-scoped Spark session that survives all tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spark():
    from src.lakehouse.spark import get_spark
    s = get_spark("CapstoneTests")
    yield s
    s.stop()


@pytest.fixture(scope="module")
def patched_spark(spark):
    """Patch get_spark to return the shared session and make stop a no-op."""
    from src.lakehouse import bronze, silver, gold, verify_delta
    original_get = None
    import src.lakehouse.spark as spark_mod

    original_get = spark_mod.get_spark
    spark_mod.get_spark = lambda *a, **kw: spark
    # Also patch the session's stop to be a no-op so build functions don't kill it.
    original_stop = spark.stop
    spark.stop = lambda: None

    yield spark

    spark_mod.get_spark = original_get
    spark.stop = original_stop


@pytest.fixture(scope="module")
def built_bronze(patched_spark):
    """Build Bronze once and reuse for Silver/Gold tests."""
    records = _sample_records(12, "healthy")
    _write_accepted(records)
    from src.lakehouse.bronze import build_bronze
    build_bronze()
    return records


class TestBronze:
    def test_bronze_created_with_correct_row_count(self, spark, built_bronze):
        df = spark.read.format("delta").load(str(config.BRONZE_PATH))
        assert df.count() == len(built_bronze)

    def test_bronze_has_expected_columns(self, spark, built_bronze):
        df = spark.read.format("delta").load(str(config.BRONZE_PATH))
        cols = set(df.columns)
        assert {"reading_id", "machine_id", "temperature_c", "recorded_at"}.issubset(cols)


class TestSilver:
    def test_silver_created_from_bronze(self, spark, built_bronze):
        from src.lakehouse.silver import build_silver
        build_silver()
        df = spark.read.format("delta").load(str(config.SILVER_PATH))
        assert df.count() == len(built_bronze)

    def test_silver_has_status_and_anomaly_flag(self, spark, built_bronze):
        from src.lakehouse.silver import build_silver
        build_silver()
        df = spark.read.format("delta").load(str(config.SILVER_PATH))
        cols = set(df.columns)
        assert "status" in cols
        assert "is_anomaly" in cols


class TestGold:
    def test_gold_is_aggregate_smaller_than_silver(self, spark, built_bronze):
        from src.lakehouse.silver import build_silver
        from src.lakehouse.gold import build_gold
        build_silver()
        build_gold()
        silver_count = spark.read.format("delta").load(str(config.SILVER_PATH)).count()
        gold_count = spark.read.format("delta").load(str(config.GOLD_PATH)).count()
        assert gold_count < silver_count, f"Gold ({gold_count}) should be smaller than Silver ({silver_count})"

    def test_gold_has_kpi_columns(self, spark, built_bronze):
        from src.lakehouse.silver import build_silver
        from src.lakehouse.gold import build_gold
        build_silver()
        build_gold()
        df = spark.read.format("delta").load(str(config.GOLD_PATH))
        cols = set(df.columns)
        expected = {"machine_id", "reading_date", "readings", "anomaly_count",
                    "avg_temperature_c", "max_temperature_c", "min_temperature_c",
                    "anomaly_rate", "temp_spread_c"}
        assert expected.issubset(cols)


class TestDeltaMerge:
    def test_merge_updates_existing_key(self, spark, built_bronze):
        """Sending the same reading_id with changed values must update, not duplicate."""
        from src.lakehouse.bronze import build_bronze
        from src.lakehouse.silver import build_silver

        # Re-publish with the overheat scenario (same reading_ids, higher temps).
        # Must append to Bronze first because Silver reads from Bronze.
        overheat = _sample_records(12, "overheat")
        _write_accepted(overheat)
        build_bronze()  # append overheat records to Bronze
        build_silver()  # MERGE from Bronze into Silver

        df = spark.read.format("delta").load(str(config.SILVER_PATH))
        assert df.count() == 12, "MERGE should not create duplicate rows"

        # Verify a specific key was actually updated.
        # CNC_MILL_01-01 is 67.0 in healthy, 97.5 in overheat.
        row = df.filter("reading_id = 'CNC_MILL_01-01'").collect()[0]
        assert row.temperature_c > 85.0, f"CNC_MILL_01-01 should have been updated to an overheat value, got {row.temperature_c}"


class TestSchemaEnforcement:
    def test_incompatible_schema_rejected(self, spark, built_bronze):
        """Delta must refuse a write with an incompatible temperature_c type."""
        from pyspark.sql.types import StringType, StructField, StructType
        from src.lakehouse.silver import build_silver
        build_silver()  # ensure Silver exists

        bad_schema = StructType([
            StructField("reading_id", StringType(), True),
            StructField("machine_id", StringType(), True),
            StructField("temperature_c", StringType(), True),  # string instead of double
            StructField("recorded_at", StringType(), True),
            StructField("reading_date", StringType(), True),
            StructField("status", StringType(), True),
            StructField("is_anomaly", StringType(), True),
        ])
        bad_df = spark.createDataFrame([(
            "SCHEMA-TEST-01", "CNC_MILL_01", "not-a-number",
            "2024-03-01 08:00:00", "2024-03-01", "NORMAL", "false"
        )], schema=bad_schema)

        with pytest.raises(Exception, match="DELTA_FAILED_TO_MERGE_FIELDS|DeltaIllegalArgumentException"):
            bad_df.write.format("delta").mode("append").save(str(config.SILVER_PATH))
