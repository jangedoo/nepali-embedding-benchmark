"""Canonical result discovery, validation, precedence, and publication."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from neb.evaluation import sha256_file
from neb.registry import load_models, load_tasks
from neb.schemas import ModelSpec, ResultRecord, RunProvenance, RuntimeSettings, VerificationStatus


def validate_run(run_dir: Path, root: Path) -> tuple[RunProvenance, list[ResultRecord]]:
    provenance_path = run_dir / "provenance.json"
    provenance = RunProvenance.model_validate_json(provenance_path.read_text(encoding="utf-8"))
    models = {item.id: item for item in load_models(root)}
    tasks = {item.id: item for item in load_tasks(root)}
    if provenance.model_id not in models or provenance.task_id not in tasks:
        raise ValueError("run references an unknown model or task")
    model, task = models[provenance.model_id], tasks[provenance.task_id]
    if (
        model.revision != provenance.model_revision
        or task.dataset.revision != provenance.dataset_revision
    ):
        raise ValueError("run revision does not match the canonical registry")
    if task.version != provenance.task_version:
        raise ValueError("run task version does not match the canonical registry")
    model_meta_path = run_dir / "model_meta.json"
    settings_path = run_dir / "run_settings.jsonl"
    if not model_meta_path.is_file() or not settings_path.is_file():
        raise ValueError("run must include model_meta.json and run_settings.jsonl")
    model_meta = ModelSpec.model_validate_json(model_meta_path.read_text(encoding="utf-8"))
    if model_meta.id != model.id or model_meta.revision != model.revision:
        raise ValueError("model_meta.json does not match the registry")
    settings_lines = [
        line for line in settings_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not settings_lines:
        raise ValueError("run_settings.jsonl is empty")
    for line in settings_lines:
        RuntimeSettings.model_validate_json(line)

    records: list[ResultRecord] = []
    seen: set[str] = set()
    for view in task.views:
        relative = f"results/{view.id}.json"
        result_path = run_dir / relative
        if not result_path.is_file():
            raise ValueError(f"missing result for {task.id}/{view.id}")
        digest = sha256_file(result_path)
        if provenance.result_hashes.get(relative) != digest:
            raise ValueError(f"result hash mismatch: {relative}")
        raw: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
        if raw.get("model_revision") != model.revision:
            raise ValueError(f"model revision mismatch: {relative}")
        if raw.get("dataset_revision") != task.dataset.revision:
            raise ValueError(f"dataset revision mismatch: {relative}")
        entries = [entry for split in raw.get("scores", {}).values() for entry in split]
        matching = [entry for entry in entries if entry.get("hf_subset") == view.id]
        if len(matching) != 1:
            raise ValueError(f"expected one score entry for {task.id}/{view.id}")
        score = matching[0].get(view.primary_metric, matching[0].get("main_score"))
        key = f"{model.revision}:{task.versioned_id}:{view.id}"
        if key in seen:
            raise ValueError(f"duplicate result: {key}")
        seen.add(key)
        records.append(
            ResultRecord(
                model_id=model.id,
                model_revision=model.revision,
                task_id=task.id,
                task_version=task.version,
                view_id=view.id,
                metric=view.primary_metric,
                score=score,
                status=provenance.status,
                result_path=relative,
                result_sha256=digest,
                dataset_revision=task.dataset.revision,
            )
        )
    return provenance, records


def publish_run(
    run_dir: Path,
    status: VerificationStatus,
    root: Path,
    *,
    skip_existing: bool = False,
) -> Path:
    provenance, _ = validate_run(run_dir, root)
    if status == VerificationStatus.verified and provenance.model_hf_id.startswith("community/"):
        raise ValueError("community placeholder models cannot be published as verified")
    destination = (
        root
        / "results"
        / status.value
        / provenance.model_id
        / provenance.model_revision
        / f"{provenance.task_id}-v{provenance.task_version}"
    )
    if destination.exists():
        if skip_existing:
            validate_run(destination, root)
            return destination
        raise FileExistsError(f"a {status.value} submission already exists: {destination}")
    shutil.copytree(run_dir, destination)
    copied = RunProvenance.model_validate_json(
        (destination / "provenance.json").read_text(encoding="utf-8")
    )
    copied.status = status
    (destination / "provenance.json").write_text(
        copied.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return destination


def discover_results(root: Path, *, include_community: bool = True) -> list[ResultRecord]:
    records: dict[tuple[str, str, str], ResultRecord] = {}
    statuses = [VerificationStatus.community, VerificationStatus.verified]
    if not include_community:
        statuses = [VerificationStatus.verified]
    for status in statuses:  # verified is deliberately later and wins
        base = root / "results" / status.value
        if not base.exists():
            continue
        for provenance_path in sorted(base.glob("*/*/*/provenance.json")):
            _, run_records = validate_run(provenance_path.parent, root)
            for record in run_records:
                record.status = status
                record.result_path = str(
                    (provenance_path.parent / record.result_path).relative_to(root)
                )
                records[(record.model_id, record.task_id, record.view_id)] = record
    return sorted(records.values(), key=lambda item: (item.task_id, item.view_id, item.model_id))
