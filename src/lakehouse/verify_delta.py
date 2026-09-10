"""Executable proofs for the two Delta requirements the rubric tests explicitly.

1. MERGE / UPSERT on a business key
   Re-send an existing reading_id with a changed temperature and prove the
   existing Silver row is UPDATED, not duplicated.

2. SCHEMA ENFORCEMENT
   Attempt an incompatible write (temperature_c as a string) and prove Delta
   refuses it and leaves rows, schema and version untouched.

Run:  python -m src.lakehouse.verify_delta
"""
from __future__ import annotations

from datetime import datetime, timezone

from delta.tables import DeltaTable
from pyspark.errors import AnalysisException
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType, DateType, DoubleType, StringType, StructField, StructType, TimestampType,
)

from src.common import config
from src.lakehouse.silver import BUSINESS_KEY, upsert_silver
from src.lakehouse.spark import get_spark

SILVER_SCHEMA = StructType([
    StructField(BUSINESS_KEY, StringType(), False),
    StructField("machine_id", StringType(), False),
    StructField("temperature_c", DoubleType(), False),
    StructField("recorded_at", TimestampType(), False),
    StructField("reading_date", DateType(), False),
    StructField("status", StringType(), False),
    StructField("is_anomaly", BooleanType(), False),
    StructField("ingested_at", TimestampType(), False),
])


def test_merge_upsert(spark) -> None:
    """Same business key + changed value must UPDATE, not duplicate."""
    print("\n=== TEST 1: Delta MERGE / UPSERT on business key ===")
    path = str(config.SILVER_PATH)
    silver = spark.read.format("delta").load(path)

    target_key = "CNC_MILL_01-00"
    before_row = silver.filter(F.col(BUSINESS_KEY) == target_key).first()
    before_total = silver.count()
    before_key_count = silver.filter(F.col(BUSINESS_KEY) == target_key).count()
    print(f"BEFORE: total_rows={before_total}, rows_for_{target_key}={before_key_count}, "
          f"temperature_c={before_row['temperature_c']}, status={before_row['status']}")

    # A corrected reading arrives for a key that already exists: 66.0 -> 91.25,
    # which also flips the derived status from NORMAL to ANOMALY.
    new_temp = 91.25
    recorded = before_row["recorded_at"]
    update = spark.createDataFrame(
        [(
            target_key, "CNC_MILL_01", new_temp, recorded,
            recorded.date(), "ANOMALY", True, datetime.now(timezone.utc),
        )],
        SILVER_SCHEMA,
    )

    result = upsert_silver(spark, update)

    after = spark.read.format("delta").load(path)
    after_row = after.filter(F.col(BUSINESS_KEY) == target_key).first()
    after_total = after.count()
    after_key_count = after.filter(F.col(BUSINESS_KEY) == target_key).count()
    print(f"AFTER:  total_rows={after_total}, rows_for_{target_key}={after_key_count}, "
          f"temperature_c={after_row['temperature_c']}, status={after_row['status']}")

    assert result["operation"] == "merge", "Expected a MERGE, not a create"
    assert after_key_count == 1, f"MERGE duplicated the key ({after_key_count} rows)"
    assert after_total == before_total, (
        f"MERGE changed the row count {before_total} -> {after_total}; it should update in place"
    )
    assert after_row["temperature_c"] == new_temp, "MERGE did not update temperature_c"
    assert after_row["status"] == "ANOMALY", "MERGE did not update the derived status"
    print(f"PASS: MERGE updated {target_key} in place "
          f"({before_row['temperature_c']} -> {new_temp}); no duplicate row created.")

    # Restore the original value so repeated runs stay deterministic.
    restore = spark.createDataFrame(
        [(
            target_key, "CNC_MILL_01", before_row["temperature_c"], recorded,
            recorded.date(), before_row["status"], before_row["is_anomaly"],
            datetime.now(timezone.utc),
        )],
        SILVER_SCHEMA,
    )
    upsert_silver(spark, restore)
    print(f"[cleanup] restored {target_key} to {before_row['temperature_c']}C")


def test_schema_enforcement(spark) -> None:
    """An incompatible write must be refused by Delta."""
    print("\n=== TEST 2: Delta schema enforcement (incompatible write) ===")
    path = str(config.SILVER_PATH)
    table = DeltaTable.forPath(spark, path)

    before_rows = spark.read.format("delta").load(path).orderBy(BUSINESS_KEY).collect()
    before_schema = spark.read.format("delta").load(path).schema.json()
    before_version = table.history(1).first()["version"]
    print(f"BEFORE: rows={len(before_rows)}, version={before_version}")

    # temperature_c as a STRING conflicts with the table's DoubleType column.
    bad_schema = StructType([
        StructField(f.name, StringType() if f.name == "temperature_c" else f.dataType, f.nullable)
        for f in SILVER_SCHEMA.fields
    ])
    now = datetime.now(timezone.utc)
    bad = spark.createDataFrame(
        [(
            "BAD-SCHEMA-001", "CNC_MILL_01", "not-a-number", now,
            now.date(), "NORMAL", False, now,
        )],
        bad_schema,
    )

    try:
        bad.write.format("delta").mode("append").save(path)
    except AnalysisException as err:
        # Only the expected schema-merge failure is tolerated; anything else is a real bug.
        if "DELTA_FAILED_TO_MERGE_FIELDS" not in str(err):
            raise
        first_line = str(err).strip().splitlines()[0]
        print(f"EXPECTED REJECTION: {first_line}")
    else:
        raise AssertionError("Delta accepted an incompatible schema - enforcement is broken!")

    after = spark.read.format("delta").load(path)
    after_version = DeltaTable.forPath(spark, path).history(1).first()["version"]
    assert after.orderBy(BUSINESS_KEY).collect() == before_rows, "Rejected write changed the rows"
    assert after.schema.json() == before_schema, "Rejected write changed the schema"
    assert after_version == before_version, (
        f"Rejected write created a new version ({before_version} -> {after_version})"
    )
    assert after.filter(F.col(BUSINESS_KEY) == "BAD-SCHEMA-001").count() == 0, \
        "Bad row leaked into the table"
    print(f"AFTER:  rows={after.count()}, version={after_version}")
    print("PASS: write refused; rows, schema and version all unchanged.")


def main() -> None:
    spark = get_spark("CapstoneVerifyDelta")
    try:
        test_merge_upsert(spark)
        test_schema_enforcement(spark)
        print("\nSUCCESS: Delta MERGE and schema enforcement both verified")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
