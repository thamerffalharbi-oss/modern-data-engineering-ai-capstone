"""Pydantic data contract enforced at the Kafka ingestion boundary.

Anything that does not satisfy this contract never reaches Bronze; it is routed
to the real Kafka dead-letter topic with the validation error attached.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common import config


class MachineReading(BaseModel):
    """One temperature reading emitted by a factory machine.

    extra="forbid" makes unknown fields a contract violation rather than
    something silently dropped into the lakehouse.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reading_id: str = Field(min_length=1)
    machine_id: str
    temperature_c: float
    recorded_at: datetime

    @field_validator("machine_id")
    @classmethod
    def machine_known(cls, v: str) -> str:
        if v not in config.VALID_MACHINES:
            raise ValueError(f"unknown machine_id {v!r}; expected one of {config.VALID_MACHINES}")
        return v

    @field_validator("temperature_c")
    @classmethod
    def temperature_plausible(cls, v: float) -> float:
        # A factory sensor outside this band is a broken sensor, not a hot machine.
        if not -50.0 <= v <= 200.0:
            raise ValueError(f"temperature_c {v} outside plausible range [-50, 200]")
        return v

    @property
    def is_anomaly(self) -> bool:
        return self.temperature_c > config.ANOMALY_THRESHOLD_C
