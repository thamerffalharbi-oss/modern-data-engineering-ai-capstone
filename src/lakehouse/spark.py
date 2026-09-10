"""Spark session factory configured for real Delta Lake.

Reuses the JDK 17 / Hadoop winutils / Ivy cache that day01_mini_lakehouse
already provisioned, including the Windows-specific fixes discovered there
(ASCII digits for the _delta_log filenames, local SPARK_LOCAL_IP).
"""
from __future__ import annotations

import os
import sys

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from src.common import config


def _prepare_windows_runtime() -> None:
    """Point Spark at the bundled JDK/Hadoop and force an ASCII numeric locale."""
    runtime = config.DAY01_RUNTIME
    if not runtime.exists():
        return

    jdk = next((p for p in runtime.iterdir() if p.is_dir() and p.name.startswith("jdk-17")), None)
    if jdk and not os.environ.get("JAVA_HOME"):
        os.environ["JAVA_HOME"] = str(jdk)
        os.environ["PATH"] = f"{jdk / 'bin'}{os.pathsep}{os.environ['PATH']}"

    hadoop = runtime / "hadoop"
    if hadoop.exists() and not os.environ.get("HADOOP_HOME"):
        os.environ["HADOOP_HOME"] = str(hadoop)
        os.environ["PATH"] = f"{hadoop / 'bin'}{os.pathsep}{os.environ['PATH']}"

    # Delta needs ASCII digits in _delta_log/NNNN.json regardless of Windows locale.
    opts = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if "user.language" not in opts:
        os.environ["JAVA_TOOL_OPTIONS"] = (
            f"{opts} -Duser.language=en -Duser.country=US "
            "-Duser.language.format=en -Duser.country.format=US"
        ).strip()


def get_spark(app_name: str = "SdaiaCapstone") -> SparkSession:
    _prepare_windows_runtime()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    builder = (
        SparkSession.builder.master(config.SPARK_MASTER)
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.showConsoleProgress", "false")
        # Schema enforcement must stay ON: an incompatible write has to fail.
        .config("spark.databricks.delta.schema.autoMerge.enabled", "false")
        .config("spark.jars.ivy", str(config.DAY01_RUNTIME / "ivy"))
        .config("spark.sql.warehouse.dir", str(config.DATA_DIR / "warehouse"))
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
