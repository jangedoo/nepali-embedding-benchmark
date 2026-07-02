"""Registry loading and cross-file policy validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from neb.schemas import ModelSpec, TaskSpec

T = TypeVar("T", bound=BaseModel)


def project_root(start: Path | None = None) -> Path:
    """Locate the checkout containing registries/, falling back to the current directory."""
    cursor = (start or Path.cwd()).resolve()
    for candidate in (cursor, *cursor.parents):
        if (candidate / "registries").is_dir():
            return candidate
    return cursor


def _load_many(directory: Path, schema: type[T]) -> list[T]:
    if not directory.exists():
        raise FileNotFoundError(f"registry directory does not exist: {directory}")
    items: list[T] = []
    for path in sorted((*directory.glob("*.yaml"), *directory.glob("*.yml"))):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            try:
                items.append(schema.model_validate(value))
            except Exception as exc:
                raise ValueError(f"invalid registry entry in {path}: {exc}") from exc
    return items


def _unique(items: Iterable[T], attribute: str) -> list[T]:
    values = list(items)
    seen: set[str] = set()
    for item in values:
        key = str(getattr(item, attribute))
        if key in seen:
            raise ValueError(f"duplicate {attribute}: {key}")
        seen.add(key)
    return values


def load_tasks(root: Path | None = None) -> list[TaskSpec]:
    directory = project_root(root) / "registries/tasks"
    if not directory.is_dir():
        directory = Path(__file__).with_name("registries") / "tasks"
    return _unique(_load_many(directory, TaskSpec), "id")


def load_models(root: Path | None = None) -> list[ModelSpec]:
    directory = project_root(root) / "registries/models"
    if not directory.is_dir():
        directory = Path(__file__).with_name("registries") / "models"
    return _unique(_load_many(directory, ModelSpec), "id")


def validate_registries(root: Path | None = None) -> tuple[list[TaskSpec], list[ModelSpec]]:
    tasks = load_tasks(root)
    models = load_models(root)
    if not tasks:
        raise ValueError("task registry is empty")
    if not models:
        raise ValueError("model registry is empty")
    return tasks, models
