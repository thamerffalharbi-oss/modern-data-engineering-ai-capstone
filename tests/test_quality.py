"""Tests for the Great Expectations quality gate: pass and fail paths."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.common import config
from src.quality.gate import (
    QualityGateFailed,
    apply_expectations,
    build_validator,
    run_gate,
)


def _gold_df(scenario: str = "healthy") -> pd.DataFrame:
    """Build a synthetic Gold DataFrame matching the real schema."""
    if scenario == "healthy":
        return pd.DataFrame([
            {"machine_id": "CNC_MILL_01", "reading_date": pd.Timestamp("2024-03-01"),
             "readings": 4, "anomaly_count": 1, "avg_temperature_c": 75.0,
             "max_temperature_c": 96.5, "min_temperature_c": 66.0,
             "anomaly_rate": 0.25, "temp_spread_c": 30.5},
            {"machine_id": "ROBOTIC_ARM_02", "reading_date": pd.Timestamp("2024-03-01"),
             "readings": 4, "anomaly_count": 1, "avg_temperature_c": 76.0,
             "max_temperature_c": 97.5, "min_temperature_c": 67.0,
             "anomaly_rate": 0.25, "temp_spread_c": 30.5},
        ])
    else:
        return pd.DataFrame([
            {"machine_id": "CNC_MILL_01", "reading_date": pd.Timestamp("2024-03-01"),
             "readings": 4, "anomaly_count": 3, "avg_temperature_c": 90.0,
             "max_temperature_c": 99.5, "min_temperature_c": 67.0,
             "anomaly_rate": 0.75, "temp_spread_c": 32.5},
        ])


class TestQualityGatePass:
    def test_healthy_gold_passes_all_expectations(self):
        df = _gold_df("healthy")
        validator = build_validator(df)
        apply_expectations(validator)
        result = validator.validate()
        assert result.success, "healthy Gold should pass all expectations"
        stats = result.statistics
        assert stats["successful_expectations"] == stats["evaluated_expectations"]


class TestQualityGateFail:
    def test_overheat_gold_fails_anomaly_rate_expectation(self):
        df = _gold_df("overheat")
        validator = build_validator(df)
        apply_expectations(validator)
        result = validator.validate()
        assert not result.success, "overheat Gold should fail the gate"
        # The anomaly_rate expectation should be among the failures.
        failed_types = [r.expectation_config.expectation_type for r in result.results if not r.success]
        assert "expect_column_values_to_be_between" in failed_types

    def test_run_gate_raises_on_failure(self):
        """run_gate must raise QualityGateFailed when Gold breaches the ceiling."""
        # Patch load_gold to return the overheat DataFrame.
        import src.quality.gate as gate
        original = gate.load_gold
        gate.load_gold = lambda: _gold_df("overheat")
        try:
            with pytest.raises(QualityGateFailed, match="anomaly_rate"):
                run_gate(save_evidence=False)
        finally:
            gate.load_gold = original
