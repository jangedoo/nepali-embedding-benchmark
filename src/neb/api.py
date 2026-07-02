"""Stable Python API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neb.registry import load_models, load_tasks, project_root
from neb.schemas import ModelSpec, RuntimeSettings, TaskSpec


@dataclass(frozen=True)
class Benchmark:
    tasks: tuple[TaskSpec, ...]
    models: tuple[ModelSpec, ...]


def get_tasks(root: Path | None = None) -> list[TaskSpec]:
    return load_tasks(root)


def get_models(root: Path | None = None) -> list[ModelSpec]:
    return load_models(root)


def get_benchmark(root: Path | None = None) -> Benchmark:
    return Benchmark(tuple(get_tasks(root)), tuple(get_models(root)))


def evaluate(
    model: str,
    tasks: list[str] | None = None,
    *,
    runtime: RuntimeSettings | None = None,
    allow_remote_code: bool = False,
    output_dir: Path | None = None,
    root: Path | None = None,
) -> list[Path]:
    """Evaluate a registered model and return produced run directories."""
    from neb.evaluation import EvaluationRunner

    runner = EvaluationRunner(project_root(root))
    return runner.run(
        model,
        tasks,
        runtime=runtime or RuntimeSettings(),
        allow_remote_code=allow_remote_code,
        output_dir=output_dir,
    )
