"""Validated registry and result schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
MetricName = Annotated[str, Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_]*$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdapterKind(str, Enum):
    sts = "sts"
    retrieval = "retrieval"
    reranking = "reranking"
    pair_classification = "pair_classification"
    bitext_mining = "bitext_mining"


class VerificationStatus(str, Enum):
    community = "community"
    verified = "verified"


class DatasetRef(StrictModel):
    id: str = Field(pattern=r"^[^/]+/[^/]+$")
    revision: SHA
    url: HttpUrl | None = None


class TaskView(StrictModel):
    id: Identifier
    split: str
    config: str | None = None
    resources: dict[str, str] = Field(default_factory=dict)
    languages: list[str] = Field(min_length=1)
    columns: dict[str, str]
    metrics: list[MetricName] = Field(min_length=1)
    primary_metric: MetricName
    description: str | None = None

    @model_validator(mode="after")
    def validate_metrics(self) -> TaskView:
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("task view metrics must be unique")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary_metric must be present in metrics")
        return self


class TaskSpec(StrictModel):
    id: Identifier
    version: int = Field(ge=1)
    display_name: str
    description: str
    dataset: DatasetRef
    adapter: AdapterKind
    views: list[TaskView] = Field(min_length=1)
    license: str = "unknown"
    homepage: HttpUrl | None = None
    reviewed_adapter: str | None = None

    @model_validator(mode="after")
    def unique_views(self) -> TaskSpec:
        ids = [view.id for view in self.views]
        if len(ids) != len(set(ids)):
            raise ValueError("task view ids must be unique")
        return self

    @property
    def versioned_id(self) -> str:
        return f"{self.id}@{self.version}"


class PromptOverrides(StrictModel):
    query: str | None = None
    document: str | None = None


class ModelSpec(StrictModel):
    id: Identifier
    display_name: str
    hf_id: str = Field(pattern=r"^[^/]+/[^/]+$")
    revision: SHA
    trust_remote_code: bool = False
    prompts: PromptOverrides = Field(default_factory=PromptOverrides)
    languages: list[str] = Field(default_factory=list)
    base_model: str | None = None
    base_model_revision: SHA | None = None
    excluded_base_model: str | None = None
    license: str = "unknown"
    homepage: HttpUrl | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def enforce_remote_code_owner(self) -> ModelSpec:
        if self.trust_remote_code and not self.hf_id.startswith("jangedoo/"):
            raise ValueError("trust_remote_code is restricted to pinned jangedoo/* models")
        if bool(self.base_model) != bool(self.base_model_revision):
            raise ValueError("base_model and base_model_revision must be set together")
        return self


class RuntimeSettings(StrictModel):
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1)
    dtype: str | None = None
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    resume: bool = False


class RunProvenance(StrictModel):
    schema_version: Literal[2] = 2
    run_id: Identifier
    status: VerificationStatus
    model_id: Identifier
    model_hf_id: str
    model_revision: SHA
    task_id: Identifier
    task_version: int = Field(ge=1)
    dataset_revision: SHA
    mteb_version: Literal["2.16.2"] = "2.16.2"
    neb_version: str
    sentence_transformers_version: str
    parameter_count: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    effective_prompts: PromptOverrides = Field(default_factory=PromptOverrides)
    runtime: RuntimeSettings
    hardware: dict[str, str] = Field(default_factory=dict)
    command: list[str] = Field(default_factory=list)
    result_hashes: dict[str, SHA256] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResultRecord(StrictModel):
    model_id: Identifier
    model_revision: SHA
    task_id: Identifier
    task_version: int = Field(ge=1)
    view_id: Identifier
    metrics: dict[str, float] = Field(min_length=1)
    status: VerificationStatus
    result_path: str
    result_sha256: SHA256
    dataset_revision: SHA
    parameter_count: int = Field(gt=0)
    vocab_size: int = Field(gt=0)

    @model_validator(mode="after")
    def finite_metrics(self) -> ResultRecord:
        for name, value in self.metrics.items():
            if not name:
                raise ValueError("metric names must not be empty")
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"metric {name!r} must be finite")
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"metric {name!r} must be in [-1, 1]")
        return self
