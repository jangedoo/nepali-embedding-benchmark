"""Deterministic v3 exports from native MTEB evidence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from neb.results import discover_results
from neb.tasks import get_tasks


def _display_name(name: str) -> str:
    return re.sub(r"\.v\d+$", "", name).replace("_", " ")


def _task_payload(task: Any) -> dict[str, Any]:
    metadata = task.metadata
    languages = metadata.eval_langs
    if isinstance(languages, list):
        subset_languages = {"default": languages}
    else:
        subset_languages = dict(languages)
    return {
        "name": metadata.name,
        "display_name": _display_name(metadata.name),
        "description": metadata.description,
        "type": metadata.type,
        "main_score": metadata.main_score,
        "dataset": {
            "name": metadata.dataset["path"],
            "revision": metadata.revision,
            "url": f"https://huggingface.co/datasets/{metadata.dataset['path']}/tree/{metadata.revision}",
        },
        "splits": list(metadata.eval_splits),
        "subsets": [
            {"name": subset, "languages": subset_languages[subset]} for subset in task.hf_subsets
        ],
    }


def export_static(root: Path, output: Path | None = None) -> list[Path]:
    destination = output or root / "site/public/data/v3"
    destination.mkdir(parents=True, exist_ok=True)
    tasks = [_task_payload(task) for task in get_tasks()]
    active_task_names = {task["name"] for task in tasks}
    records = [record for record in discover_results(root) if record.task_name in active_task_names]
    results = [record.model_dump(mode="json", exclude={"model_metadata"}) for record in records]

    models: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record.model_name, record.model_revision)
        current = models.get(key)
        evaluated = record.evaluated_at or ""
        if current is None or evaluated > (current["evaluated_at"] or ""):
            metadata = record.model_metadata
            models[key] = {
                "name": record.model_name,
                "repository": record.model_name,
                "revision": record.model_revision,
                "evaluated_at": record.evaluated_at,
                "status": record.status.value,
                "effective_prompts": record.effective_prompts,
                "n_parameters": metadata.get("n_parameters", "unknown"),
                "embed_dim": metadata.get("embed_dim", "unknown"),
            }
        elif record.status.value == "verified":
            current["status"] = "verified"

    model_values = sorted(models.values(), key=lambda row: (row["repository"], row["revision"]))
    repositories = {model["repository"] for model in model_values}
    latest = {
        repository: max(
            (model for model in model_values if model["repository"] == repository),
            key=lambda model: (model["evaluated_at"] or "", model["revision"]),
        )["revision"]
        for repository in repositories
    }
    for model in model_values:
        model["is_latest"] = latest.get(model["repository"]) == model["revision"]

    payloads = {
        "tasks.json": {"schema_version": 3, "tasks": tasks},
        "models.json": {"schema_version": 3, "models": model_values},
        "results.json": {"schema_version": 3, "results": results},
        "catalog.json": {
            "schema_version": 3,
            "counts": {"tasks": len(tasks), "models": len(model_values), "results": len(results)},
            "tasks": tasks,
            "models": model_values,
            "results": results,
        },
    }
    written: list[Path] = []
    for filename, payload in payloads.items():
        path = destination / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)

    csv_path = destination / "results.csv"
    fields = [
        "model_name",
        "model_revision",
        "task_name",
        "task_type",
        "split",
        "subset",
        "metric",
        "score",
        "is_main_score",
        "dataset_name",
        "dataset_revision",
        "mteb_version",
        "status",
        "result_path",
        "result_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            base = record.model_dump(
                mode="json",
                exclude={
                    "metrics",
                    "languages",
                    "main_score",
                    "main_score_name",
                    "effective_prompts",
                    "evaluated_at",
                    "model_metadata",
                },
            )
            for metric, score in sorted(record.metrics.items()):
                writer.writerow(
                    {
                        **base,
                        "metric": metric,
                        "score": score,
                        "is_main_score": metric == record.main_score_name,
                    }
                )
    written.append(csv_path)
    return written
