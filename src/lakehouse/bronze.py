"""BRONZE layer - raw contract-accepted Kafka readings, appended as-is.

Bronze keeps the ingestion payload plus Kafka provenance (partition/offset) and
an ingestion timestamp. No business logic, no deduplication: it is the replayable
landing zone.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from pyspark.sql.types import (
    BooleanType, DoubleType, IntegerType, LongType, StringType, StructField, StructType, TimestampType,
)

from src.common import config
from src.lakehouse.spark import get_spark
from src.lineage.emitter import lineage_stage

BRONZE_SCHEMA = StructType([
    StructField("reading_id", StringType(), False),
    StructField("machine_id", StringType(), False),
    StructField("temperature_c", DoubleType(), False),
    StructField("recorded_at", TimestampType(), False),
    StructField("is_anomaly", BooleanType(), False),
    StructField("kafka_partition", IntegerType(), False),
    StructField("kafka_offset", LongType(), False),
    StructField("ingested_at", TimestampType(), False),
])


def load_accepted() -> list[dict]:
    if not config.INGEST_ACCEPTED.exists():
        raise FileNotFoundError(
            f"{config.INGEST_ACCEPTED} not found - run the ingestion stage first."
        )
    text = config.INGEST_ACCEPTED.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("No accepted records to load into Bronze.")
    return [json.loads(line) for line in text.splitlines()]


def build_bronze() -> int:
    records = load_accepted()
    now = datetime.now(timezone.utc)
    rows = [
        (
            r["reading_id"],
            r["machine_id"],
            float(r["temperature_c"]),
            datetime.fromisoformat(r["recorded_at"]),
            bool(r["is_anomaly"]),
            int(r["kafka_partition"]),
            int(r["kafka_offset"]),
            now,
        )
        for r in records
    ]

    spark = get_spark("CapstoneBronze")
    try:
        df = spark.createDataFrame(rows, BRONZE_SCHEMA)
        # append: Bronze is an immutable, replayable log of what Kafka delivered.
        df.write.format("delta").mode("append").save(str(config.BRONZE_PATH))
        total = spark.read.format("delta").load(str(config.BRONZE_PATH)).count()
        print(f"[bronze] appended {len(rows)} rows -> {config.BRONZE_PATH}")
        print(f"[bronze] table now holds {total} rows")
        return len(rows)
    finally:
        spark.stop()


def main() -> None:
    with lineage_stage("capstone.bronze"):
        build_bronze()


if __name__ == "__main__":
    main()
