from __future__ import annotations

import pytest
from pydantic import ValidationError

from neb.schemas import ModelSpec, ResultRecord, TaskSpec

SHA = "a" * 40


def test_task_requires_unique_views() -> None:
    data = {
        "id": "task",
        "version": 1,
        "display_name": "Task",
        "description": "Description",
        "dataset": {"id": "owner/data", "revision": SHA},
        "adapter": "sts",
        "views": [
            {
                "id": "same",
                "split": "test",
                "languages": ["ne"],
                "columns": {},
                "primary_metric": "f1",
            },
            {
                "id": "same",
                "split": "test",
                "languages": ["en"],
                "columns": {},
                "primary_metric": "f1",
            },
        ],
    }
    with pytest.raises(ValidationError, match="unique"):
        TaskSpec.model_validate(data)


def test_remote_code_is_owner_only() -> None:
    with pytest.raises(ValidationError, match="restricted"):
        ModelSpec(
            id="unsafe",
            display_name="Unsafe",
            hf_id="other/model",
            revision=SHA,
            trust_remote_code=True,
        )


def test_result_rejects_out_of_range_and_bad_hash() -> None:
    with pytest.raises(ValidationError):
        ResultRecord(
            model_id="model",
            model_revision=SHA,
            task_id="task",
            task_version=1,
            view_id="view",
            metric="f1",
            score=1.1,
            status="verified",
            result_path="result.json",
            result_sha256="b" * 64,
            dataset_revision=SHA,
        )
