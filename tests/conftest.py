"""Shared pytest fixtures for the capstone test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the capstone package importable when tests run from the capstone/ root.
CAPSTONE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPSTONE_ROOT))

# Use a temporary data directory so tests never clobber real evidence.
# Set env vars BEFORE any src.* module is imported so config picks them up.
TMP_DATA = CAPSTONE_ROOT / "data_test"
os.environ["CAPSTONE_DATA_DIR"] = str(TMP_DATA)
os.environ["CAPSTONE_EVIDENCE_DIR"] = str(TMP_DATA / "evidence")
os.environ["CAPSTONE_LINEAGE_DIR"] = str(TMP_DATA / "evidence" / "lineage")
os.environ["CAPSTONE_BRONZE_PATH"] = str(TMP_DATA / "bronze" / "readings")
os.environ["CAPSTONE_SILVER_PATH"] = str(TMP_DATA / "silver" / "readings")
os.environ["CAPSTONE_GOLD_PATH"] = str(TMP_DATA / "gold" / "machine_daily_kpi")
os.environ["CAPSTONE_INGEST_ACCEPTED"] = str(TMP_DATA / "ingest" / "accepted.jsonl")
os.environ["CAPSTONE_INGEST_REJECTED"] = str(TMP_DATA / "ingest" / "rejected.jsonl")

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cleanup_data_dir():
    """Remove the throwaway data directory after the test session."""
    yield
    import shutil
    if TMP_DATA.exists():
        shutil.rmtree(TMP_DATA, ignore_errors=True)
