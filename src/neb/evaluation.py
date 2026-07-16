"""Thin execution wrapper around MTEB 2.18.3."""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mteb
from mteb.models import ModelMeta
from mteb.results import ModelResult

from neb.models import resolve_model
from neb.tasks import get_tasks

MTEB_VERSION = "2.18.3"

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum(path: Path) -> Path:
    checksum = path.with_suffix(path.suffix + ".sha256")
    checksum.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return checksum


def _select_tasks(names: Sequence[str] | None) -> list[mteb.AbsTask]:
    tasks = get_tasks()
    if not names:
        return tasks
    by_name = {task.metadata.name: task for task in tasks}
    aliases = {re.sub(r"\.v\d+$", "", task.metadata.name): task for task in tasks}
    selected = []
    for name in names:
        task = by_name.get(name) or aliases.get(name)
        if task is None:
            choices = ", ".join(by_name)
            raise ValueError(f"unknown task {name!r}; choose one of: {choices}")
        selected.append(task)
    return selected


def _load_model(
    meta: ModelMeta,
    *,
    device: str,
    dtype: str | None,
    query_prompt: str | None,
    document_prompt: str | None,
) -> Any:
    loader_kwargs = dict(meta.loader_kwargs)
    if dtype is not None:
        aliases = {
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp16": "float16",
            "float16": "float16",
            "fp32": "float32",
            "float32": "float32",
        }
        try:
            resolved_dtype = aliases[dtype.lower()]
        except KeyError as exc:
            raise ValueError("dtype must be bf16, fp16, or fp32") from exc
        model_kwargs = dict(loader_kwargs.get("model_kwargs", {}))
        model_kwargs["torch_dtype"] = resolved_dtype
        loader_kwargs["model_kwargs"] = model_kwargs
    meta = meta.model_copy(update={"loader_kwargs": loader_kwargs}, deep=True)
    loader_name = getattr(meta.loader, "__name__", str(meta.loader))
    logger.info(
        "Loading model name=%s revision=%s loader=%s device=%s dtype=%s",
        meta.name,
        meta.revision,
        loader_name,
        device,
        dtype or "default",
    )
    model = meta.load_model(device=device)

    native_prompts = dict(getattr(model, "model_prompts", None) or {})
    configured = dict(loader_kwargs.get("model_prompts") or {})
    effective_prompts = {**native_prompts, **configured}
    if query_prompt is not None:
        effective_prompts["query"] = query_prompt
    if document_prompt is not None:
        effective_prompts["document"] = document_prompt
    if hasattr(model, "model_prompts"):
        model.model_prompts = effective_prompts
    wrapped = getattr(model, "model", None)
    if wrapped is not None and hasattr(wrapped, "prompts"):
        wrapped.prompts = effective_prompts
        default_prompt_name = getattr(wrapped, "default_prompt_name", None)
        if default_prompt_name is not None:
            logger.info(
                "Native SentenceTransformer default prompt: name=%s value=%r",
                default_prompt_name,
                effective_prompts.get(default_prompt_name),
            )

    logger.info("Effective MTEB model prompts: %s", effective_prompts)

    saved_kwargs = dict(model.mteb_model_meta.loader_kwargs)
    saved_kwargs["model_prompts"] = effective_prompts
    model.mteb_model_meta = model.mteb_model_meta.model_copy(
        update={"loader_kwargs": saved_kwargs}, deep=True
    )
    return model


def evaluate(
    model: str | ModelMeta,
    revision: str | None = None,
    tasks: Sequence[str] | Sequence[mteb.AbsTask] | None = None,
    *,
    cache_path: Path | str = "runs",
    device: str = "cpu",
    batch_size: int = 32,
    dtype: str | None = None,
    allow_remote_code: bool = False,
    query_prompt: str | None = None,
    document_prompt: str | None = None,
    show_progress_bar: bool = True,
    encode_kwargs: dict[str, Any] | None = None,
) -> ModelResult:
    """Evaluate through MTEB, preserving its ``ModelResult`` and task JSON."""
    if importlib.metadata.version("mteb") != MTEB_VERSION:
        raise RuntimeError(f"NEB requires exactly mteb {MTEB_VERSION}")
    if isinstance(model, str):
        meta = resolve_model(
            model,
            revision,
            allow_remote_code=allow_remote_code,
            query_prompt=query_prompt,
            document_prompt=document_prompt,
        )
    else:
        meta = model
        if meta.revision is None:
            raise ValueError("model metadata must contain a revision")

    if tasks and not isinstance(tasks[0], str):
        selected = list(tasks)  # type: ignore[arg-type]
    else:
        selected = _select_tasks(tasks)  # type: ignore[arg-type]
    logger.info(
        "Evaluation settings: tasks=%s device=%s batch_size=%d dtype=%s cache=%s",
        [task.metadata.name for task in selected],
        device,
        batch_size,
        dtype or "default",
        cache_path,
    )
    loaded_model = _load_model(
        meta,
        device=device,
        dtype=dtype,
        query_prompt=query_prompt,
        document_prompt=document_prompt,
    )
    cache = mteb.ResultCache(cache_path=cache_path)
    kwargs = {"batch_size": batch_size, **(encode_kwargs or {})}
    result = mteb.evaluate(
        loaded_model,
        selected,
        cache=cache,
        overwrite_strategy="only-missing",
        encode_kwargs=kwargs,
        show_progress_bar=show_progress_bar,
        co2_tracker=False,
    )
    for task_result in result.task_results:
        path = cache.get_task_result_path(
            task_result.task_name, result.model_name, result.model_revision
        )
        if path.is_file():
            write_checksum(path)
    return result
