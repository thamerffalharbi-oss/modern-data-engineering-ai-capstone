"""Tests for OpenLineage event emission: START, COMPLETE, FAIL."""
from __future__ import annotations

import json

import pytest

from src.lineage.emitter import lineage_stage, events_file, RunState
from openlineage.client.event_v2 import RunEvent


def _read_events() -> list[dict]:
    f = events_file()
    if not f.exists():
        return []
    return [json.loads(l) for l in f.open(encoding="utf-8")]


class TestLineageEvents:
    def test_start_and_complete_emitted(self):
        with lineage_stage("test.stage.success") as run_id:
            pass  # no exception -> COMPLETE

        events = _read_events()
        recent = [e for e in events if e["job"]["name"] == "test.stage.success"]
        states = [e["eventType"] for e in recent]
        assert "START" in states
        assert "COMPLETE" in states

    def test_fail_emitted_on_exception(self):
        with pytest.raises(ValueError, match="deliberate"):
            with lineage_stage("test.stage.fail"):
                raise ValueError("deliberate failure")

        events = _read_events()
        recent = [e for e in events if e["job"]["name"] == "test.stage.fail"]
        states = [e["eventType"] for e in recent]
        assert "START" in states
        assert "FAIL" in states

        # The FAIL event should carry an errorMessage facet.
        fail_event = next(e for e in recent if e["eventType"] == "FAIL")
        facets = fail_event["run"]["facets"]
        assert "errorMessage" in facets
        assert "deliberate" in facets["errorMessage"]["message"]

    def test_events_are_openlineage_compliant(self):
        """Every event must have the required OpenLineage RunEvent fields."""
        with lineage_stage("test.stage.schema"):
            pass

        events = _read_events()
        for e in events:
            if e["job"]["name"] != "test.stage.schema":
                continue
            assert "eventType" in e
            assert "eventTime" in e
            assert "run" in e
            assert "runId" in e["run"]
            assert "job" in e
            assert "namespace" in e["job"]
            assert "name" in e["job"]
            assert "producer" in e
