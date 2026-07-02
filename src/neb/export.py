"""Deterministic dashboard exports generated from canonical source data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from neb.registry import load_models, load_tasks
from neb.results import discover_model_metadata, discover_results


def _jsonable(items: list[Any]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def export_static(root: Path, output: Path | None = None) -> list[Path]:
    destination = output or root / "site/public/data/v2"
    destination.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(root)
    models = load_models(root)
    results = discover_results(root)
    metadata = discover_model_metadata(root)
    exported_models = []
    for model in models:
        values = metadata.get((model.id, model.revision))
        exported_models.append(
            {
                **model.model_dump(mode="json"),
                "parameter_count": values[0] if values else "unknown",
                "vocab_size": values[1] if values else "unknown",
            }
        )
    payloads = {
        "tasks.json": {"schema_version": 2, "tasks": _jsonable(tasks)},
        "models.json": {"schema_version": 2, "models": exported_models},
        "results.json": {"schema_version": 2, "results": _jsonable(results)},
        "catalog.json": {
            "schema_version": 2,
            "counts": {"tasks": len(tasks), "models": len(models), "results": len(results)},
            "tasks": _jsonable(tasks),
            "models": exported_models,
            "results": _jsonable(results),
        },
    }
    written: list[Path] = []
    for name, payload in payloads.items():
        path = destination / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    csv_path = destination / "results.csv"
    fields = [
        "model_id",
        "model_revision",
        "task_id",
        "task_version",
        "view_id",
        "metric",
        "score",
        "is_primary",
        "status",
        "dataset_revision",
        "parameter_count",
        "vocab_size",
        "result_sha256",
        "result_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        views = {(task.id, view.id): view for task in tasks for view in task.views}
        for record in results:
            values = record.model_dump(mode="json")
            metrics = values.pop("metrics")
            view = views[(record.task_id, record.view_id)]
            for metric in view.metrics:
                writer.writerow(
                    {
                        **values,
                        "metric": metric,
                        "score": metrics[metric],
                        "is_primary": metric == view.primary_metric,
                    }
                )
    written.append(csv_path)
    return written
