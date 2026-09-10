"""Central configuration for the capstone pipeline.

Every value can be overridden through an environment variable so the same code
runs on a laptop, in Airflow, and in CI. No secrets are stored here - the LLM
credentials are read from the environment only (see src/rag/generate.py).
"""
import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------
# CAPSTONE_ROOT is the `capstone/` directory (this file is src/common/config.py).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("CAPSTONE_DATA_DIR", ROOT / "data"))
EVIDENCE_DIR = Path(os.getenv("CAPSTONE_EVIDENCE_DIR", ROOT / "evidence"))
LINEAGE_DIR = Path(os.getenv("CAPSTONE_LINEAGE_DIR", EVIDENCE_DIR / "lineage"))

# Delta lakehouse layers.
BRONZE_PATH = Path(os.getenv("CAPSTONE_BRONZE_PATH", DATA_DIR / "bronze" / "readings"))
SILVER_PATH = Path(os.getenv("CAPSTONE_SILVER_PATH", DATA_DIR / "silver" / "readings"))
GOLD_PATH = Path(os.getenv("CAPSTONE_GOLD_PATH", DATA_DIR / "gold" / "machine_daily_kpi"))

# Where the ingestion stage parks accepted / rejected records for the next stage.
INGEST_ACCEPTED = Path(os.getenv("CAPSTONE_INGEST_ACCEPTED", DATA_DIR / "ingest" / "accepted.jsonl"))
INGEST_REJECTED = Path(os.getenv("CAPSTONE_INGEST_REJECTED", DATA_DIR / "ingest" / "rejected.jsonl"))

# --- Kafka ---------------------------------------------------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_READINGS = os.getenv("KAFKA_TOPIC_READINGS", "capstone.readings")
TOPIC_DLQ = os.getenv("KAFKA_TOPIC_DLQ", "capstone.dlq")
CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "capstone-ingest")

# --- Spark / Delta -------------------------------------------------------
# Reuse the JDK/Hadoop/Ivy runtime that day01 already provisioned.
DAY01_RUNTIME = ROOT.parent / "day01_mini_lakehouse" / ".runtime"
SPARK_MASTER = os.getenv("CAPSTONE_SPARK_MASTER", "local[2]")

# --- Quality gate --------------------------------------------------------
# Anomaly rate above this fraction fails the Great Expectations gate.
MAX_ANOMALY_RATE = float(os.getenv("CAPSTONE_MAX_ANOMALY_RATE", "0.60"))
ANOMALY_THRESHOLD_C = float(os.getenv("CAPSTONE_ANOMALY_THRESHOLD_C", "85.0"))
VALID_MACHINES = ["CNC_MILL_01", "ROBOTIC_ARM_02", "ASSEMBLY_LINE_03"]

# --- RAG -----------------------------------------------------------------
EMBED_MODEL = os.getenv("CAPSTONE_EMBED_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("CAPSTONE_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Reuse the model cache day03 already downloaded so tests run offline.
HF_CACHE = Path(os.getenv("HF_HOME", ROOT.parent / "day03_rag_pipeline" / ".hf_cache"))


def ensure_dirs() -> None:
    """Create the directories the pipeline writes to."""
    for p in (DATA_DIR, EVIDENCE_DIR, LINEAGE_DIR, INGEST_ACCEPTED.parent):
        p.mkdir(parents=True, exist_ok=True)
