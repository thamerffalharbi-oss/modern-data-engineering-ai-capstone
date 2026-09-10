"""Real Kafka consumer that enforces the Pydantic contract at the boundary.

Valid records  -> written to data/ingest/accepted.jsonl (input to Bronze).
Invalid records -> produced to the REAL Kafka dead-letter topic (capstone.dlq)
                   with the payload, the rejection reason, the validation error
                   detail and a timestamp; also mirrored to rejected.jsonl so
                   the evidence survives after topic retention expires.

An invalid record never reaches the accepted stream, so it cannot enter Bronze.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from confluent_kafka import Consumer, KafkaError, Producer
from pydantic import ValidationError

from src.common import config
from src.ingestion.contract import MachineReading
from src.lineage.emitter import lineage_stage


def build_consumer() -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": config.KAFKA_BOOTSTRAP,
            "group.id": config.CONSUMER_GROUP,
            # Read the whole demo batch from the beginning on a fresh group.
            "auto.offset.reset": "earliest",
            # Commit only after a record has been routed (accepted or DLQ'd).
            "enable.auto.commit": False,
        }
    )


def build_dlq_producer() -> Producer:
    return Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP, "acks": "all"})


def _dlq_envelope(raw: bytes, reason: str, detail: object) -> dict:
    return {
        "rejected_at": datetime.now(timezone.utc).isoformat(),
        "source_topic": config.TOPIC_READINGS,
        "rejection_reason": reason,
        "validation_error": detail,
        "original_payload": raw.decode("utf-8", errors="replace"),
    }


def consume_once(max_idle_polls: int = 8, poll_timeout: float = 1.0) -> dict:
    """Drain the readings topic once, routing each record. Returns a summary."""
    config.ensure_dirs()
    consumer = build_consumer()
    dlq = build_dlq_producer()
    consumer.subscribe([config.TOPIC_READINGS])

    accepted: list[dict] = []
    rejected: list[dict] = []
    idle = 0

    try:
        while idle < max_idle_polls:
            msg = consumer.poll(poll_timeout)
            if msg is None:
                idle += 1
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    idle += 1
                    continue
                raise RuntimeError(f"Kafka consume error: {msg.error()}")

            idle = 0
            raw = msg.value()

            # --- Contract enforcement at the ingestion boundary -----------
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                env = _dlq_envelope(raw, "malformed_json", str(exc))
                dlq.produce(config.TOPIC_DLQ, value=json.dumps(env).encode())
                rejected.append(env)
                consumer.commit(msg, asynchronous=False)
                continue

            try:
                reading = MachineReading(**payload)
            except ValidationError as exc:
                env = _dlq_envelope(
                    raw,
                    "schema_validation_failed",
                    [
                        {"field": ".".join(str(p) for p in e["loc"]), "error": e["msg"]}
                        for e in exc.errors()
                    ],
                )
                dlq.produce(config.TOPIC_DLQ, value=json.dumps(env).encode())
                rejected.append(env)
                print(f"[consumer] REJECTED {payload.get('reading_id', '<no id>')} -> "
                      f"{config.TOPIC_DLQ}: {env['validation_error']}")
                consumer.commit(msg, asynchronous=False)
                continue

            record = reading.model_dump(mode="json")
            record["is_anomaly"] = reading.is_anomaly
            record["kafka_offset"] = msg.offset()
            record["kafka_partition"] = msg.partition()
            accepted.append(record)
            print(f"[consumer] ACCEPTED {reading.reading_id} "
                  f"({reading.machine_id} {reading.temperature_c}C)")
            consumer.commit(msg, asynchronous=False)
    finally:
        dlq.flush(30)
        consumer.close()

    config.INGEST_ACCEPTED.write_text(
        "\n".join(json.dumps(r) for r in accepted), encoding="utf-8"
    )
    config.INGEST_REJECTED.write_text(
        "\n".join(json.dumps(r) for r in rejected), encoding="utf-8"
    )

    return {"accepted": len(accepted), "rejected": len(rejected)}


def main() -> None:
    with lineage_stage("capstone.ingestion"):
        summary = consume_once()
        print(f"\n[consumer] accepted={summary['accepted']} "
              f"rejected_to_dlq={summary['rejected']}")
        print(f"[consumer] accepted -> {config.INGEST_ACCEPTED}")
        print(f"[consumer] rejected -> {config.INGEST_REJECTED} (and topic {config.TOPIC_DLQ})")
        if summary["accepted"] == 0:
            raise RuntimeError("Ingestion produced no valid records; refusing to continue.")


if __name__ == "__main__":
    main()
