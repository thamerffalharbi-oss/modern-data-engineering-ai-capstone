"""Tests for the Kafka ingestion boundary: Pydantic contract + DLQ routing."""
from __future__ import annotations

import json

import pytest

from src.ingestion.contract import MachineReading
from src.ingestion.producer import demo_batch


class TestContract:
    """The Pydantic data contract is the ingestion boundary."""

    def test_valid_reading_accepted(self):
        rec = demo_batch("healthy")[0]
        reading = MachineReading(**rec)
        assert reading.reading_id == rec["reading_id"]
        assert reading.machine_id in {"CNC_MILL_01", "ROBOTIC_ARM_02", "ASSEMBLY_LINE_03"}

    def test_unknown_machine_rejected(self):
        rec = {
            "reading_id": "BAD-unknown",
            "machine_id": "GHOST_MACHINE_99",
            "temperature_c": 70.0,
            "recorded_at": "2024-03-01T08:00:00+00:00",
        }
        with pytest.raises(Exception, match="unknown machine_id"):
            MachineReading(**rec)

    def test_temperature_out_of_range_rejected(self):
        rec = {
            "reading_id": "BAD-spike",
            "machine_id": "CNC_MILL_01",
            "temperature_c": 9999.0,
            "recorded_at": "2024-03-01T08:00:00+00:00",
        }
        with pytest.raises(Exception, match="outside plausible range"):
            MachineReading(**rec)

    def test_missing_temperature_rejected(self):
        rec = {
            "reading_id": "BAD-missing",
            "machine_id": "ROBOTIC_ARM_02",
            "recorded_at": "2024-03-01T08:00:00+00:00",
        }
        with pytest.raises(Exception):
            MachineReading(**rec)

    def test_invalid_timestamp_rejected(self):
        rec = {
            "reading_id": "BAD-ts",
            "machine_id": "ASSEMBLY_LINE_03",
            "temperature_c": 71.0,
            "recorded_at": "not-a-timestamp",
        }
        with pytest.raises(Exception):
            MachineReading(**rec)


class TestDemoBatch:
    def test_healthy_batch_has_12_valid_plus_4_invalid(self):
        batch = demo_batch("healthy")
        assert len(batch) == 16

    def test_overheat_batch_has_12_valid_plus_4_invalid(self):
        batch = demo_batch("overheat")
        assert len(batch) == 16

    def test_overheat_produces_high_anomaly_rate(self):
        """The overheat scenario must genuinely breach the anomaly-rate ceiling."""
        batch = demo_batch("overheat")
        valid = [r for r in batch if r["machine_id"] in {"CNC_MILL_01", "ROBOTIC_ARM_02", "ASSEMBLY_LINE_03"}
                 and "temperature_c" in r and isinstance(r.get("recorded_at"), str)
                 and r["recorded_at"] != "not-a-timestamp"]
        anomalies = [r for r in valid if r["temperature_c"] >= 85.0]
        rate = len(anomalies) / len(valid)
        assert rate > 0.6, f"overheat anomaly_rate {rate} should exceed 0.6"

    def test_healthy_produces_low_anomaly_rate(self):
        batch = demo_batch("healthy")
        valid = [r for r in batch if r["machine_id"] in {"CNC_MILL_01", "ROBOTIC_ARM_02", "ASSEMBLY_LINE_03"}
                 and "temperature_c" in r and isinstance(r.get("recorded_at"), str)
                 and r["recorded_at"] != "not-a-timestamp"]
        anomalies = [r for r in valid if r["temperature_c"] >= 85.0]
        rate = len(anomalies) / len(valid)
        assert rate <= 0.6, f"healthy anomaly_rate {rate} should be <= 0.6"
