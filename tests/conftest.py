from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mteb.results import TaskResult

from neb.evaluation import MTEB_VERSION, write_checksum
from neb.tasks import STSB_REVISION


def make_sts_cache(
    root: Path,
    *,
    score: float = 0.5,
    model: str = "owner/model",
    revision: str = "a" * 40,
    subset: str = "ne-ne",
    loader_kwargs: dict[str, object] | None = None,
    batch_size: int = 2,
) -> Path:
    directory = root / "results" / model.replace("/", "__") / revision
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "STSBNepali.v3.json"
    TaskResult(
        dataset_revision=STSB_REVISION,
        task_name="STSBNepali.v3",
        mteb_version=MTEB_VERSION,
        scores={
            "test": [
                {
                    "hf_subset": subset,
                    "languages": ["nep-Deva"],
                    "mteb_version": MTEB_VERSION,
                    "cosine_spearman": score,
                    "cosine_pearson": score - 0.1,
                    "main_score": score,
                }
            ]
        },
        evaluation_time=1.0,
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ).to_disk(path)
    (directory / "model_meta.json").write_text(
        json.dumps(
            {
                "name": model,
                "revision": revision,
                "loader_kwargs": loader_kwargs or {"model_prompts": {"query": "query: "}},
                "n_parameters": 10,
                "embed_dim": 4,
            }
        ),
        encoding="utf-8",
    )
    (directory / "run_settings.jsonl").write_text(
        json.dumps(
            {
                "task": "STSBNepali.v3",
                "split": "test",
                "subset": subset,
                "version": {"mteb": MTEB_VERSION},
                "encode_kwargs": {"batch_size": batch_size},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_checksum(path)
    return path
