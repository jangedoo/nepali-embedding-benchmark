"""Deterministic dashboard exports generated from canonical source data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from neb.registry import load_models, load_tasks
from neb.results import discover_results


def _jsonable(items: list[Any]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def export_static(root: Path, output: Path | None = None) -> list[Path]:
    destination = output or root / "site/public/data/v1"
    destination.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(root)
    models = load_models(root)
    results = discover_results(root)
    payloads = {
        "tasks.json": {"schema_version": 1, "tasks": _jsonable(tasks)},
        "models.json": {"schema_version": 1, "models": _jsonable(models)},
        "results.json": {"schema_version": 1, "results": _jsonable(results)},
        "catalog.json": {
            "schema_version": 1,
            "counts": {"tasks": len(tasks), "models": len(models), "results": len(results)},
            "tasks": _jsonable(tasks),
            "models": _jsonable(models),
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
        "status",
        "dataset_revision",
        "result_sha256",
        "result_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in results:
            writer.writerow(record.model_dump(mode="json"))
    written.append(csv_path)
    return written
