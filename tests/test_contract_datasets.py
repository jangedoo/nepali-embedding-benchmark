from __future__ import annotations

import os

import pytest

from neb.tasks import NANOBEIR_SUBSETS, get_tasks

pytestmark = pytest.mark.skipif(
    os.environ.get("NEB_CONTRACT_TESTS") != "1",
    reason="set NEB_CONTRACT_TESTS=1 for pinned network contracts",
)


def test_pinned_hugging_face_contracts() -> None:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    for task in get_tasks()[:5]:
        path, revision = task.metadata.dataset["path"], task.metadata.revision
        assert len(revision) == 40
        if task.metadata.name == "NanoBEIRNepaliRetrieval.v3":
            assert set(get_dataset_config_names(path, revision=revision)) >= {
                "corpus",
                "queries",
                "qrels",
            }
            assert set(get_dataset_split_names(path, "corpus", revision=revision)) >= set(
                NANOBEIR_SUBSETS.values()
            )
        else:
            sample = load_dataset(path, split="test[:1]", revision=revision)
            assert len(sample) == 1
