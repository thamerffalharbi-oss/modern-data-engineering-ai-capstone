"""GOLD layer - a real analytical aggregate, not a copy of Silver.

Grain: one row per (machine_id, reading_date) - Silver's grain is one row per
reading, so Gold genuinely reduces the data. Business KPIs produced:

  readings          - how many readings that machine reported that day
  anomaly_count     - how many exceeded the temperature threshold
  anomaly_rate      - anomaly_count / readings  (the KPI the quality gate checks)
  avg_temperature_c - mean temperature
  max_temperature_c - peak temperature
  min_temperature_c - lowest temperature
  temp_spread_c     - max - min, i.e. thermal volatility for the day

Gold is rebuilt with mode("overwrite") because it is a pure derivation of Silver.
"""
from __future__ import annotations

from pyspark.sql import functions as F

from src.common import config
from src.lakehouse.spark import get_spark
from src.lineage.emitter import lineage_stage


def build_gold() -> int:
    spark = get_spark("CapstoneGold")
    try:
        silver = spark.read.format("delta").load(str(config.SILVER_PATH))

        gold = (
            silver.groupBy("machine_id", "reading_date")
            .agg(
                F.count("*").alias("readings"),
                F.sum(F.col("is_anomaly").cast("int")).alias("anomaly_count"),
                F.round(F.avg("temperature_c"), 2).alias("avg_temperature_c"),
                F.round(F.max("temperature_c"), 2).alias("max_temperature_c"),
                F.round(F.min("temperature_c"), 2).alias("min_temperature_c"),
            )
            .withColumn(
                "anomaly_rate",
                F.round(F.col("anomaly_count") / F.col("readings"), 4),
            )
            .withColumn(
                "temp_spread_c",
                F.round(F.col("max_temperature_c") - F.col("min_temperature_c"), 2),
            )
            .orderBy("machine_id", "reading_date")
        )

        gold.write.format("delta").mode("overwrite").save(str(config.GOLD_PATH))

        rows = spark.read.format("delta").load(str(config.GOLD_PATH))
        n = rows.count()
        silver_n = silver.count()
        print(f"[gold] aggregated {silver_n} Silver rows -> {n} Gold rows "
              f"(grain: machine_id x reading_date)")
        rows.orderBy("machine_id", "reading_date").show(truncate=False)

        # Gold must be a real reduction, never a Silver copy.
        if n >= silver_n:
            raise AssertionError(
                f"Gold ({n} rows) did not aggregate Silver ({silver_n} rows); "
                "Gold must not be a copy of Silver."
            )
        print(f"[gold] OK: Gold is a genuine aggregate ({silver_n} -> {n} rows)")
        return n
    finally:
        spark.stop()


def main() -> None:
    with lineage_stage("capstone.gold"):
        build_gold()


if __name__ == "__main__":
    main()
