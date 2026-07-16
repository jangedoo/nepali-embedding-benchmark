"""Schemas for NEB model overrides, evidence, and the static v3 export."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class VerificationStatus(str, Enum):
    community = "community"
    verified = "verified"


class ModelOverride(BaseModel):
    """Exact-revision behavior that is missing from upstream model metadata."""

    model_config = ConfigDict(extra="forbid")

    hf_id: str = Field(pattern=r"^[^/]+/[^/]+$")
    revision: SHA
    prompts: dict[str, str] = Field(default_factory=dict)
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def validate_remote_code_owner(self) -> ModelOverride:
        if self.trust_remote_code and not self.hf_id.startswith("jangedoo/"):
            raise ValueError("trust_remote_code is restricted to exact jangedoo/* revisions")
        if any(not key or not isinstance(value, str) for key, value in self.prompts.items()):
            raise ValueError("prompt keys must be non-empty and prompt values must be strings")
        return self


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_revision: SHA
    task_name: str
    task_type: str
    split: str
    subset: str
    languages: list[str]
    metrics: dict[str, float]
    main_score_name: str
    main_score: float
    dataset_name: str
    dataset_revision: SHA
    mteb_version: str
    status: VerificationStatus
    result_path: str
    result_sha256: SHA256
    effective_prompts: dict[str, str]
    evaluated_at: str | None = None
    model_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def finite_metrics(self) -> EvidenceRecord:
        for name, value in self.metrics.items():
            if not name or not isinstance(value, (int, float)):
                raise ValueError("metrics must have names and numeric values")
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"metric {name!r} must be finite")
        if self.main_score_name not in self.metrics:
            raise ValueError("main score must be one of the exported scalar metrics")
        if self.metrics[self.main_score_name] != self.main_score:
            raise ValueError("main score value does not match its named metric")
        return self
