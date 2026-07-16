"""Validation, publication, and precedence for native MTEB evidence."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from mteb.results import TaskResult

from neb.evaluation import MTEB_VERSION, sha256_file, write_checksum
from neb.schemas import EvidenceRecord, VerificationStatus
from neb.tasks import get_result_tasks


def _checksum_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _read_checksum(path: Path) -> str:
    checksum_path = _checksum_path(path)
    if not checksum_path.is_file():
        raise ValueError(f"missing checksum: {checksum_path}")
    parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != path.name:
        raise ValueError(f"malformed checksum: {checksum_path}")
    if parts[0] != sha256_file(path):
        raise ValueError(f"checksum mismatch: {path}")
    return parts[0]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _tasks_by_name() -> dict[str, Any]:
    return {task.metadata.name: task for task in get_result_tasks()}


def _validate_model_meta(path: Path) -> dict[str, Any]:
    meta_path = path.parent / "model_meta.json"
    if not meta_path.is_file():
        raise ValueError("native cache entry is missing model_meta.json")
    meta = _load_json(meta_path)
    name, revision = meta.get("name"), meta.get("revision")
    if (
        isinstance(name, str)
        and name.startswith("local/")
        or isinstance(revision, str)
        and revision.startswith("local-")
    ):
        raise ValueError("local model results cannot be published as canonical NEB evidence")
    if not isinstance(name, str) or "/" not in name:
        raise ValueError("model_meta.json has no Hugging Face model name")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("model_meta.json has no exact model revision")
    if path.parent.name != revision:
        raise ValueError("model revision does not match its cache directory")
    expected_model_dir = name.replace("/", "__").replace(" ", "_")
    if path.parent.parent.name != expected_model_dir:
        raise ValueError("model name does not match its cache directory")
    loader_kwargs = meta.get("loader_kwargs")
    if not isinstance(loader_kwargs, dict) or not isinstance(
        loader_kwargs.get("model_prompts"), dict
    ):
        raise ValueError("model metadata must record effective prompts")
    return meta


def _validate_settings(path: Path, result: TaskResult) -> None:
    settings_path = path.parent / "run_settings.jsonl"
    if not settings_path.is_file():
        raise ValueError("native cache entry is missing run_settings.jsonl")
    entries = []
    for line in settings_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("run_settings.jsonl contains malformed JSON") from exc
    if not entries:
        raise ValueError("run_settings.jsonl is empty")
    keys = {(item.get("task"), item.get("split"), item.get("subset")) for item in entries}
    for split, scores in result.scores.items():
        for score in scores:
            key = (result.task_name, split, score["hf_subset"])
            if key not in keys:
                raise ValueError(f"missing run settings for {key}")
    for item in entries:
        if item.get("task") != result.task_name:
            continue
        versions = item.get("version")
        if not isinstance(versions, dict) or versions.get("mteb") != MTEB_VERSION:
            raise ValueError("run settings have an incompatible MTEB version")


def validate_task_result(
    path: Path,
    *,
    status: VerificationStatus = VerificationStatus.community,
    root: Path | None = None,
) -> tuple[TaskResult, dict[str, Any], list[EvidenceRecord]]:
    """Validate one MTEB task JSON and return scalar evidence rows."""
    try:
        result = TaskResult.from_disk(path)
    except Exception as exc:
        raise ValueError(f"invalid MTEB TaskResult: {path}") from exc
    task = _tasks_by_name().get(result.task_name)
    if task is None:
        raise ValueError(f"unknown NEB task: {result.task_name}")
    if path.name != f"{result.task_name}.json":
        raise ValueError("task result filename does not match task_name")
    if result.dataset_revision != task.metadata.revision:
        raise ValueError("dataset revision does not match the task definition")
    if result.mteb_version != MTEB_VERSION:
        raise ValueError("task result has an incompatible MTEB version")
    digest = _read_checksum(path)
    meta = _validate_model_meta(path)
    _validate_settings(path, result)

    main_metric = task.metadata.main_score
    records: list[EvidenceRecord] = []
    seen: set[tuple[str, str]] = set()
    for split, entries in result.scores.items():
        if split not in task.metadata.eval_splits:
            raise ValueError(f"unexpected split {split!r} for {result.task_name}")
        for entry in entries:
            subset = entry.get("hf_subset")
            if subset not in task.hf_subsets:
                raise ValueError(f"unexpected subset {subset!r} for {result.task_name}")
            key = (split, subset)
            if key in seen:
                raise ValueError(f"duplicate score entry for {split}/{subset}")
            seen.add(key)
            if entry.get("mteb_version") != MTEB_VERSION:
                raise ValueError("subset score has an incompatible MTEB version")
            metrics = {
                name: float(value)
                for name, value in entry.items()
                if name not in {"hf_subset", "languages", "mteb_version", "main_score"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            }
            if main_metric not in metrics:
                raise ValueError(f"missing main metric {main_metric!r}")
            main_score = entry.get("main_score")
            if (
                not isinstance(main_score, (int, float))
                or float(main_score) != metrics[main_metric]
            ):
                raise ValueError("main_score does not match the task main metric")
            result_path = (
                str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
            )
            records.append(
                EvidenceRecord(
                    model_name=meta["name"],
                    model_revision=meta["revision"],
                    task_name=result.task_name,
                    task_type=task.metadata.type,
                    split=split,
                    subset=subset,
                    languages=list(entry.get("languages", [])),
                    metrics=metrics,
                    main_score_name=main_metric,
                    main_score=float(main_score),
                    dataset_name=task.metadata.dataset["path"],
                    dataset_revision=result.dataset_revision,
                    mteb_version=MTEB_VERSION,
                    status=status,
                    result_path=result_path,
                    result_sha256=digest,
                    effective_prompts=meta["loader_kwargs"]["model_prompts"],
                    evaluated_at=result.date.isoformat() if result.date else None,
                    model_metadata=meta,
                )
            )
    if not records:
        raise ValueError("task result contains no score entries")
    return result, meta, records


def _task_jsons(source: Path) -> list[Path]:
    if source.is_file():
        if source.name == "model_meta.json":
            raise ValueError("model_meta.json is metadata, not an MTEB task result")
        return [source]
    direct = [path for path in source.glob("*.json") if path.name != "model_meta.json"]
    nested = [path for path in source.glob("results/*/*/*.json") if path.name != "model_meta.json"]
    paths = sorted({*direct, *nested})
    if not paths:
        raise ValueError(f"no native MTEB task results found under {source}")
    return paths


def _score_map(result: TaskResult) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (split, entry["hf_subset"]): entry
        for split, entries in result.scores.items()
        for entry in entries
    }


def _result_values_equal(existing: Any, incoming: Any) -> bool:
    if isinstance(existing, float) and isinstance(incoming, float):
        if math.isnan(existing) and math.isnan(incoming):
            return True
    if isinstance(existing, dict) and isinstance(incoming, dict):
        return existing.keys() == incoming.keys() and all(
            _result_values_equal(existing[key], incoming[key]) for key in existing
        )
    if isinstance(existing, list) and isinstance(incoming, list):
        return len(existing) == len(incoming) and all(
            _result_values_equal(left, right)
            for left, right in zip(existing, incoming, strict=True)
        )
    return existing == incoming


def _merge_results(
    existing: TaskResult, incoming: TaskResult, *, overwrite: bool = False
) -> TaskResult:
    if (
        existing.task_name != incoming.task_name
        or existing.dataset_revision != incoming.dataset_revision
    ):
        raise ValueError("cannot merge different tasks or dataset revisions")
    current = _score_map(existing)
    additions = _score_map(incoming)
    for key in current.keys() & additions.keys():
        if not _result_values_equal(current[key], additions[key]) and not overwrite:
            differing_fields = sorted(
                field
                for field in current[key].keys() | additions[key].keys()
                if not _result_values_equal(current[key].get(field), additions[key].get(field))
            )
            raise ValueError(
                f"conflicting published scores for {key[0]}/{key[1]}; "
                f"differing fields: {', '.join(differing_fields)}; "
                "pass --overwrite to replace them"
            )
    scores = {
        split: [additions.get((split, entry["hf_subset"]), entry) for entry in entries]
        for split, entries in existing.scores.items()
    }
    for (split, _), entry in additions.items():
        if (split, entry["hf_subset"]) not in current:
            scores.setdefault(split, []).append(entry)
    return existing.model_copy(
        update={
            "scores": scores,
            "evaluation_time": max(existing.evaluation_time or 0, incoming.evaluation_time or 0),
            "date": max(filter(None, [existing.date, incoming.date]), default=None),
        }
    )


def _settings_key(item: dict[str, Any]) -> tuple[Any, Any, Any]:
    return item.get("task"), item.get("split"), item.get("subset")


def _read_settings_lines(path: Path) -> list[tuple[str, dict[str, Any]]]:
    lines: list[tuple[str, dict[str, Any]]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"run_settings.jsonl contains malformed JSON: {path}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"run_settings.jsonl contains a non-object entry: {path}")
        lines.append((raw, item))
    return lines


def _merge_settings_lines(
    existing: list[tuple[str, dict[str, Any]]],
    incoming: list[tuple[str, dict[str, Any]]],
    *,
    overwrite: bool,
) -> list[str]:
    merged = list(existing)
    positions = {_settings_key(item): index for index, (_, item) in enumerate(merged)}
    for raw, item in incoming:
        key = _settings_key(item)
        position = positions.get(key)
        if position is None:
            positions[key] = len(merged)
            merged.append((raw, item))
            continue
        if merged[position][1] == item:
            continue
        if not overwrite:
            raise ValueError(
                f"conflicting run settings for {key[0]}/{key[1]}/{key[2]}; "
                "pass --overwrite to replace them"
            )
        merged[position] = (raw, item)
    return [raw for raw, _ in merged]


def publish_results(
    source: Path,
    status: VerificationStatus,
    root: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Publish native caches, optionally replacing colliding canonical evidence."""
    candidates: list[dict[str, Any]] = []
    for source_path in _task_jsons(source):
        incoming, meta, _ = validate_task_result(source_path, status=status)
        model_dir = meta["name"].replace("/", "__").replace(" ", "_")
        destination = (
            root
            / "results"
            / status.value
            / "results"
            / model_dir
            / meta["revision"]
            / source_path.name
        )
        destination_meta = destination.parent / "model_meta.json"
        source_meta = source_path.parent / "model_meta.json"
        source_meta_data = _load_json(source_meta)
        destination_meta_data = _load_json(destination_meta) if destination_meta.exists() else None
        if destination_meta_data is not None and destination_meta_data != source_meta_data:
            if not overwrite:
                raise ValueError(
                    "conflicting model metadata for published revision; "
                    "pass --overwrite with a complete revision cache to replace it"
                )
        if destination.exists():
            existing, _, _ = validate_task_result(destination, status=status, root=root)
            merged = _merge_results(existing, incoming, overwrite=overwrite)
        else:
            merged = incoming
        candidates.append(
            {
                "source": source_path,
                "incoming": incoming,
                "meta": source_meta_data,
                "source_meta": source_meta,
                "destination": destination,
                "destination_meta": destination_meta,
                "destination_meta_data": destination_meta_data,
                "merged": merged,
            }
        )

    groups: dict[Path, list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(candidate["destination"].parent, []).append(candidate)

    group_settings: dict[Path, list[str]] = {}
    for directory, group in groups.items():
        source_metas = [candidate["meta"] for candidate in group]
        if any(meta != source_metas[0] for meta in source_metas[1:]):
            raise ValueError(f"source contains conflicting model metadata for {directory.name}")

        metadata_changed = any(
            candidate["destination_meta_data"] is not None
            and candidate["destination_meta_data"] != candidate["meta"]
            for candidate in group
        )
        if metadata_changed:
            incoming_by_name = {candidate["destination"].name: candidate for candidate in group}
            missing: list[str] = []
            for existing_path in sorted(directory.glob("*.json")):
                if existing_path.name == "model_meta.json":
                    continue
                candidate = incoming_by_name.get(existing_path.name)
                if candidate is None:
                    missing.append(existing_path.name)
                    continue
                existing, _, _ = validate_task_result(existing_path, status=status, root=root)
                uncovered = _score_map(existing).keys() - _score_map(candidate["incoming"]).keys()
                missing.extend(
                    f"{existing_path.name}:{split}/{subset}" for split, subset in sorted(uncovered)
                )
            if missing:
                details = ", ".join(missing)
                raise ValueError(
                    "overwriting model metadata requires source coverage for every "
                    f"published score in the revision; missing {details}"
                )

        destination_settings = directory / "run_settings.jsonl"
        existing_settings = (
            _read_settings_lines(destination_settings) if destination_settings.exists() else []
        )
        incoming_settings: list[tuple[str, dict[str, Any]]] = []
        for candidate in group:
            result_keys = {
                (candidate["incoming"].task_name, split, subset)
                for split, subset in _score_map(candidate["incoming"])
            }
            source_settings = candidate["source"].parent / "run_settings.jsonl"
            incoming_settings.extend(
                (raw, item)
                for raw, item in _read_settings_lines(source_settings)
                if _settings_key(item) in result_keys
            )
        group_settings[directory] = _merge_settings_lines(
            existing_settings, incoming_settings, overwrite=overwrite
        )

    published: list[Path] = []
    for candidate in candidates:
        source_path = candidate["source"]
        destination = candidate["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            candidate["merged"].to_disk(destination)
        else:
            shutil.copy2(source_path, destination)
        write_checksum(destination)
        published.append(destination)

    for directory, group in groups.items():
        shutil.copy2(group[0]["source_meta"], directory / "model_meta.json")
        settings = group_settings[directory]
        (directory / "run_settings.jsonl").write_text("\n".join(settings) + "\n", encoding="utf-8")
        for candidate in group:
            validate_task_result(candidate["destination"], status=status, root=root)
    return published


def discover_results(root: Path, *, include_community: bool = True) -> list[EvidenceRecord]:
    """Resolve verified precedence independently per revision/task/split/subset."""
    selected: dict[tuple[str, str, str, str, str], EvidenceRecord] = {}
    statuses = [VerificationStatus.community, VerificationStatus.verified]
    if not include_community:
        statuses = [VerificationStatus.verified]
    for status in statuses:
        base = root / "results" / status.value / "results"
        if not base.exists():
            continue
        for path in sorted(base.glob("*/*/*.json")):
            if path.name == "model_meta.json":
                continue
            _, _, records = validate_task_result(path, status=status, root=root)
            for record in records:
                key = (
                    record.model_name,
                    record.model_revision,
                    record.task_name,
                    record.split,
                    record.subset,
                )
                selected[key] = record
    return sorted(
        selected.values(),
        key=lambda row: (row.task_name, row.subset, row.model_name, row.model_revision),
    )


def validate_repository(root: Path) -> int:
    return len(discover_results(root))
