from __future__ import annotations

import os
from pathlib import Path

import pytest

from neb.registry import load_tasks

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("NEB_CONTRACT_TESTS") != "1",
    reason="set NEB_CONTRACT_TESTS=1 for pinned network contracts",
)


def test_pinned_dataset_splits_and_columns() -> None:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    for task in load_tasks(ROOT):
        if task.id == "nanobeir-ne":
            configs = get_dataset_config_names(task.dataset.id, revision=task.dataset.revision)
            assert set(configs) >= {"corpus", "queries", "qrels"}
            continue
        for view in task.views:
            splits = get_dataset_split_names(
                task.dataset.id, view.config, revision=task.dataset.revision
            )
            assert view.split in splits
            if task.adapter.value == "pair_classification":
                assert "valid" in splits
        sample = load_dataset(
            task.dataset.id,
            task.views[0].config,
            split=task.views[0].split,
            revision=task.dataset.revision,
        ).select(range(1))
        assert set(task.views[0].columns.values()) <= set(sample.column_names)
