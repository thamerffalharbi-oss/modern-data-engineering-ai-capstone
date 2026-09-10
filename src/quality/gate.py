"""Great Expectations quality gate over the Gold aggregate.

Uses the real Great Expectations engine: an EphemeralDataContext, a pandas
datasource, a named ExpectationSuite and a Validator that computes real metrics.
The gate reads the Gold Delta table and validates the business KPIs.

If validation fails the function raises QualityGateFailed. The exception is NOT
swallowed, so the Airflow task fails and every downstream task is skipped.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import great_expectations as ge
import pandas as pd

from src.common import config
from src.lineage.emitter import lineage_stage

warnings.filterwarnings("ignore", category=DeprecationWarning)

SUITE_NAME = "capstone_gold_kpi_suite"


class QualityGateFailed(RuntimeError):
    """Raised when the Great Expectations suite does not pass."""


def load_gold() -> pd.DataFrame:
    """Read the Gold Delta table into pandas for validation.

    Uses the ``deltalake`` Rust reader (no JVM / Spark executor) so the gate
    works reliably inside an Airflow subprocess where Spark's BlockManager
    registration can fail on Windows.
    """
    from deltalake import DeltaTable

    dt = DeltaTable(str(config.GOLD_PATH))
    pdf = dt.to_pandas()
    # reading_date comes back as date objects; make it a proper datetime column.
    pdf["reading_date"] = pd.to_datetime(pdf["reading_date"])
    return pdf


def build_validator(df: pd.DataFrame):
    context = ge.get_context(mode="ephemeral")
    datasource = context.sources.add_pandas("capstone_gold_source")
    asset = datasource.add_dataframe_asset("gold_machine_daily_kpi")
    batch_request = asset.build_batch_request(dataframe=df)
    suite = context.add_or_update_expectation_suite(SUITE_NAME)
    return context.get_validator(batch_request=batch_request, expectation_suite=suite)


def apply_expectations(validator) -> None:
    """Declare the expectations that define 'trustworthy Gold data'."""
    # --- Structure -------------------------------------------------------
    validator.expect_table_columns_to_match_set(
        column_set=[
            "machine_id", "reading_date", "readings", "anomaly_count",
            "avg_temperature_c", "max_temperature_c", "min_temperature_c",
            "anomaly_rate", "temp_spread_c",
        ]
    )
    validator.expect_table_row_count_to_be_between(min_value=1, max_value=10_000)

    # --- Completeness ----------------------------------------------------
    for col in ("machine_id", "reading_date", "readings", "anomaly_rate"):
        validator.expect_column_values_to_not_be_null(col)

    # --- Validity --------------------------------------------------------
    validator.expect_column_values_to_be_in_set("machine_id", value_set=config.VALID_MACHINES)
    validator.expect_column_values_to_be_of_type("readings", "int64")

    # --- Uniqueness: one KPI row per machine per day ---------------------
    validator.expect_compound_columns_to_be_unique(column_list=["machine_id", "reading_date"])

    # --- Ranges / plausibility -------------------------------------------
    validator.expect_column_values_to_be_between("readings", min_value=1, max_value=100_000)
    validator.expect_column_values_to_be_between("anomaly_count", min_value=0, max_value=100_000)
    validator.expect_column_values_to_be_between(
        "avg_temperature_c", min_value=-50.0, max_value=200.0
    )
    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="max_temperature_c", column_B="min_temperature_c", or_equal=True
    )

    # --- THE GATE: operational anomaly-rate ceiling ----------------------
    # A machine whose readings are mostly anomalies means the feed (or the
    # machine) cannot be trusted, so the batch must not reach the RAG stage.
    validator.expect_column_values_to_be_between(
        "anomaly_rate", min_value=0.0, max_value=config.MAX_ANOMALY_RATE
    )


def run_gate(save_evidence: bool = True) -> dict:
    """Validate Gold. Returns the summary, or raises QualityGateFailed."""
    df = load_gold()
    print(f"[quality] validating {len(df)} Gold rows "
          f"(max allowed anomaly_rate = {config.MAX_ANOMALY_RATE})")

    validator = build_validator(df)
    apply_expectations(validator)
    result = validator.validate()

    stats = result.statistics
    failed = [
        {
            "expectation": r.expectation_config.expectation_type,
            "column": r.expectation_config.kwargs.get("column")
                      or r.expectation_config.kwargs.get("column_list")
                      or r.expectation_config.kwargs.get("column_A"),
            "observed_value": r.result.get("observed_value"),
            "unexpected_count": r.result.get("unexpected_count"),
            "partial_unexpected_list": r.result.get("partial_unexpected_list"),
        }
        for r in result.results if not r.success
    ]

    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "success": bool(result.success),
        "expectations_evaluated": stats["evaluated_expectations"],
        "expectations_passed": stats["successful_expectations"],
        "expectations_failed": stats["unsuccessful_expectations"],
        "success_percent": round(stats["success_percent"], 2),
        "gold_rows": len(df),
        "max_anomaly_rate_allowed": config.MAX_ANOMALY_RATE,
        "observed_max_anomaly_rate": float(df["anomaly_rate"].max()),
        "failed_expectations": failed,
    }

    print(f"[quality] {summary['expectations_passed']}/{summary['expectations_evaluated']} "
          f"expectations passed ({summary['success_percent']}%)")
    print(f"[quality] observed max anomaly_rate = {summary['observed_max_anomaly_rate']}")

    if save_evidence:
        config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        out = config.EVIDENCE_DIR / "quality_gate_result.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[quality] result -> {out}")

    if not result.success:
        for f in failed:
            print(f"[quality] FAILED: {f['expectation']} on {f['column']} "
                  f"-> observed={f['observed_value']} unexpected={f['unexpected_count']} "
                  f"{f['partial_unexpected_list']}")
        # Raised on purpose: this is what stops the downstream Airflow tasks.
        raise QualityGateFailed(
            f"Great Expectations gate FAILED: "
            f"{summary['expectations_failed']} of {summary['expectations_evaluated']} "
            f"expectations did not pass (max anomaly_rate observed "
            f"{summary['observed_max_anomaly_rate']} > allowed {config.MAX_ANOMALY_RATE})"
        )

    print("[quality] PASS: Gold data satisfies every expectation")
    return summary


def main() -> None:
    with lineage_stage("capstone.quality"):
        run_gate()


if __name__ == "__main__":
    main()
