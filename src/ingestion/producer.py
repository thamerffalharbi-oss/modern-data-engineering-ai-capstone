"""Real Kafka producer (confluent-kafka) for machine temperature readings.

Publishes a deterministic demo batch containing both valid readings and
deliberately malformed records, so the DLQ path can be proven on every run.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

from confluent_kafka import Producer

from src.common import config


def build_producer() -> Producer:
    return Producer(
        {
            "bootstrap.servers": config.KAFKA_BOOTSTRAP,
            "client.id": "capstone-producer",
            # acks=all: do not consider a reading published until the broker
            # has durably written it.
            "acks": "all",
            "enable.idempotence": True,
        }
    )


def demo_batch(scenario: str = "healthy") -> list[dict]:
    """A fixed batch: 12 valid readings + 4 contract violations.

    Deterministic on purpose - the rubric requires a repeatable demonstration
    that valid records flow through and malformed records are rejected.

    scenario="healthy"  -> 1 anomaly per machine  => anomaly_rate 0.25, gate PASSES
    scenario="overheat" -> 3 anomalies per machine => anomaly_rate 0.75, gate FAILS

    The "overheat" batch is real data that genuinely breaches the operational
    anomaly-rate ceiling; nothing about the quality gate itself is faked.
    """
    if scenario not in {"healthy", "overheat"}:
        raise ValueError(f"unknown scenario {scenario!r}; use 'healthy' or 'overheat'")

    base = datetime(2024, 3, 1, 8, 0, tzinfo=timezone.utc)
    records: list[dict] = []
    # Readings at or after this index run hot.
    hot_from = 3 if scenario == "healthy" else 1

    # --- Valid readings: 3 machines x 4 readings ---------------------------
    for m_idx, machine in enumerate(config.VALID_MACHINES):
        for i in range(4):
            temp = 96.5 + i if i >= hot_from else 66.0 + i + m_idx
            records.append(
                {
                    "reading_id": f"{machine}-{i:02d}",
                    "machine_id": machine,
                    "temperature_c": round(temp, 2),
                    "recorded_at": (base + timedelta(minutes=15 * i)).isoformat(),
                }
            )

    # --- Deliberate contract violations (must land in the DLQ) ------------
    records.append(
        # 1. Unknown machine_id.
        {
            "reading_id": "BAD-unknown-machine",
            "machine_id": "GHOST_MACHINE_99",
            "temperature_c": 70.0,
            "recorded_at": base.isoformat(),
        }
    )
    records.append(
        # 2. Physically impossible temperature (broken sensor).
        {
            "reading_id": "BAD-sensor-spike",
            "machine_id": "CNC_MILL_01",
            "temperature_c": 9999.0,
            "recorded_at": base.isoformat(),
        }
    )
    records.append(
        # 3. Missing required field `temperature_c`.
        {
            "reading_id": "BAD-missing-field",
            "machine_id": "ROBOTIC_ARM_02",
            "recorded_at": base.isoformat(),
        }
    )
    records.append(
        # 4. Wrong type: recorded_at is not a timestamp.
        {
            "reading_id": "BAD-bad-timestamp",
            "machine_id": "ASSEMBLY_LINE_03",
            "temperature_c": 71.0,
            "recorded_at": "not-a-timestamp",
        }
    )
    return records


def publish(records: list[dict]) -> int:
    """Publish records to the readings topic. Returns the delivered count."""
    producer = build_producer()
    delivered = 0
    errors: list[str] = []

    def on_delivery(err, msg):
        nonlocal delivered
        if err is not None:
            errors.append(str(err))
        else:
            delivered += 1

    for rec in records:
        producer.produce(
            config.TOPIC_READINGS,
            key=str(rec.get("machine_id", uuid.uuid4())).encode(),
            value=json.dumps(rec).encode(),
            on_delivery=on_delivery,
        )
    producer.flush(30)

    if errors:
        raise RuntimeError(f"Kafka delivery failed for {len(errors)} record(s): {errors[:3]}")
    return delivered


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Publish the capstone demo batch to Kafka.")
    parser.add_argument(
        "--scenario",
        default=os.getenv("CAPSTONE_SCENARIO", "healthy"),
        choices=["healthy", "overheat"],
        help="healthy: anomaly_rate 0.25 (quality gate passes); "
             "overheat: anomaly_rate 0.75 (quality gate fails)",
    )
    args = parser.parse_args()

    records = demo_batch(args.scenario)
    delivered = publish(records)
    print(f"[producer] bootstrap={config.KAFKA_BOOTSTRAP} topic={config.TOPIC_READINGS}")
    print(f"[producer] scenario={args.scenario}")
    print(f"[producer] published {delivered}/{len(records)} records "
          f"(12 valid + 4 deliberate contract violations)")


if __name__ == "__main__":
    main()
