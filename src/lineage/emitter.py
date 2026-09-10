"""Real OpenLineage instrumentation for the capstone pipeline.

This uses the official `openlineage-python` client and emits spec-compliant
RunEvent objects (START / COMPLETE / FAIL) through OpenLineage's own
FileTransport. FileTransport is chosen because it is an officially supported
transport that needs no external collector, so the emitted events can be kept
as reproducible evidence.

Point OPENLINEAGE_URL at a Marquez/OpenLineage HTTP endpoint and set
CAPSTONE_LINEAGE_TRANSPORT=http to ship the same events to a real backend.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from openlineage.client import OpenLineageClient
from openlineage.client.event_v2 import Job, Run, RunEvent, RunState
from openlineage.client.transport.file import FileConfig, FileTransport
from openlineage.client.uuid import generate_new_uuid

from src.common import config

NAMESPACE = os.getenv("OPENLINEAGE_NAMESPACE", "sdaia-capstone")
PRODUCER = "https://github.com/SDAIAAcademy/modern-data-engineering-capstone"

# One file per pipeline run keeps evidence readable.
_EVENTS_FILE = config.LINEAGE_DIR / "openlineage_events.jsonl"


def _build_client() -> OpenLineageClient:
    """Build an OpenLineage client using an officially supported transport."""
    if os.getenv("CAPSTONE_LINEAGE_TRANSPORT", "file").lower() == "http" and os.getenv("OPENLINEAGE_URL"):
        # OpenLineageClient.from_environment() honours OPENLINEAGE_URL / API key.
        return OpenLineageClient.from_environment()

    config.LINEAGE_DIR.mkdir(parents=True, exist_ok=True)
    # append=True so every stage in a DAG run lands in the same evidence file.
    return OpenLineageClient(transport=FileTransport(FileConfig(log_file_path=str(_EVENTS_FILE), append=True)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(client: OpenLineageClient, state: RunState, job_name: str, run_id: str, facets: dict | None = None) -> None:
    event = RunEvent(
        eventType=state,
        eventTime=_now(),
        run=Run(runId=run_id, facets=facets or {}),
        job=Job(namespace=NAMESPACE, name=job_name),
        producer=PRODUCER,
    )
    client.emit(event)


@contextmanager
def lineage_stage(job_name: str, run_id: str | None = None):
    """Wrap a pipeline stage so it emits START then COMPLETE, or FAIL on error.

    The exception is always re-raised: lineage must never swallow a failure,
    otherwise the Airflow quality gate could not stop downstream tasks.
    """
    client = _build_client()
    run_id = run_id or str(generate_new_uuid())
    _emit(client, RunState.START, job_name, run_id)
    try:
        yield run_id
    except BaseException as exc:
        _emit(
            client,
            RunState.FAIL,
            job_name,
            run_id,
            facets={
                "errorMessage": {
                    "_producer": PRODUCER,
                    "_schemaURL": (
                        "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json"
                    ),
                    "message": f"{type(exc).__name__}: {exc}",
                    "programmingLanguage": "PYTHON",
                }
            },
        )
        raise
    else:
        _emit(client, RunState.COMPLETE, job_name, run_id)


def events_file() -> Path:
    return _EVENTS_FILE
