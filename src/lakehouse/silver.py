"""SILVER layer - cleaned, typed, deduplicated readings maintained by MERGE.

Silver is upserted from Bronze with a REAL Delta `DeltaTable.merge(...)` keyed on
the business key `reading_id`. Re-running the pipeline, or receiving a corrected
value for a reading that already exists, updates the row in place instead of
duplicating it.

Cleaning applied on the way in:
  * deduplicate Bronze by reading_id, keeping the latest Kafka offset (the most
    recent delivery wins);
  * derive `reading_date` and `status` used by the Gold aggregate;
  * normalise machine_id casing/whitespace.
"""
from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common import config
from src.lakehouse.spark import get_spark
from src.lineage.emitter import lineage_stage

# The business key the MERGE is keyed on.
BUSINESS_KEY = "reading_id"


def clean_from_bronze(spark: SparkSession) -> DataFrame:
    bronze = spark.read.format("delta").load(str(config.BRONZE_PATH))

    # Keep one row per business key: the latest Kafka delivery for that reading.
    latest = Window.partitionBy(BUSINESS_KEY).orderBy(
        F.col("kafka_offset").desc(), F.col("ingested_at").desc()
    )
    return (
        bronze.withColumn("_rn", F.row_number().over(latest))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .withColumn("machine_id", F.upper(F.trim(F.col("machine_id"))))
        .withColumn("temperature_c", F.round(F.col("temperature_c").cast("double"), 2))
        .withColumn("reading_date", F.to_date(F.col("recorded_at")))
        .withColumn(
            "status",
            F.when(F.col("temperature_c") > F.lit(config.ANOMALY_THRESHOLD_C), F.lit("ANOMALY"))
            .otherwise(F.lit("NORMAL")),
        )
        .select(
            BUSINESS_KEY, "machine_id", "temperature_c", "recorded_at",
            "reading_date", "status", "is_anomaly", "ingested_at",
        )
    )


def upsert_silver(spark: SparkSession, source: DataFrame) -> dict:
    """Create Silver, or MERGE into it when it already exists."""
    path = str(config.SILVER_PATH)

    if not DeltaTable.isDeltaTable(spark, path):
        source.write.format("delta").mode("errorifexists").save(path)
        count = spark.read.format("delta").load(path).count()
        print(f"[silver] created table with {count} rows -> {path}")
        return {"operation": "create", "rows": count, "merged": 0}

    target = DeltaTable.forPath(spark, path)
    before = target.toDF().count()

    # --- The real Delta MERGE / UPSERT on the business key ----------------
    (
        target.alias("t")
        .merge(source.alias("s"), f"t.{BUSINESS_KEY} = s.{BUSINESS_KEY}")
        .whenMatchedUpdate(
            set={
                "machine_id": "s.machine_id",
                "temperature_c": "s.temperature_c",
                "recorded_at": "s.recorded_at",
                "reading_date": "s.reading_date",
                "status": "s.status",
                "is_anomaly": "s.is_anomaly",
                "ingested_at": "s.ingested_at",
            }
        )
        .whenNotMatchedInsertAll()
        .execute()
    )

    after = target.toDF().count()
    metrics = target.history(1).select("operation", "operationMetrics").first()
    print(f"[silver] MERGE on {BUSINESS_KEY}: {before} rows -> {after} rows")
    print(f"[silver] delta operation={metrics['operation']} metrics={metrics['operationMetrics']}")
    return {
        "operation": "merge",
        "rows": after,
        "rows_before": before,
        "metrics": metrics["operationMetrics"],
    }


def build_silver() -> dict:
    spark = get_spark("CapstoneSilver")
    try:
        source = clean_from_bronze(spark)
        result = upsert_silver(spark, source)
        silver = spark.read.format("delta").load(str(config.SILVER_PATH))
        print(f"[silver] distinct {BUSINESS_KEY}s = {silver.select(BUSINESS_KEY).distinct().count()}")
        silver.orderBy("machine_id", "recorded_at").show(truncate=False)
        return result
    finally:
        spark.stop()


def main() -> None:
    with lineage_stage("capstone.silver"):
        build_silver()


if __name__ == "__main__":
    main()
