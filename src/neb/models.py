"""MTEB-native model resolution with exact, data-driven behavior overrides."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, get_args

import mteb
import yaml
from mteb.abstasks.task_metadata import TaskType
from mteb.models import ModelMeta
from mteb.types import PromptType

from neb.schemas import ModelOverride
from neb.tasks import get_tasks

SHA_LENGTH = 40
LOCAL_REVISION_PREFIX = "local-"

logger = logging.getLogger(__name__)


def _registry_directory(root: Path | None = None) -> Path:
    if root is not None:
        return root / "registries" / "models"
    cursor = Path.cwd().resolve()
    for candidate in (cursor, *cursor.parents):
        directory = candidate / "registries" / "models"
        if directory.is_dir():
            return directory
    return Path(__file__).with_name("registries") / "models"


def _valid_prompt_bases() -> set[str]:
    return {
        *get_args(TaskType),
        *(prompt_type.value for prompt_type in PromptType),
        *(task.metadata.name for task in get_tasks()),
    }


def _is_valid_prompt_key(key: str) -> bool:
    bases = _valid_prompt_bases()
    if key in bases:
        return True
    endings = tuple(f"-{prompt_type.value}" for prompt_type in PromptType)
    if key.endswith(endings):
        return key.rsplit("-", 1)[0] in bases
    return False


def _filter_prompts(
    prompts: dict[str, str] | None,
    *,
    source: str,
    strict: bool = False,
) -> dict[str, str]:
    valid: dict[str, str] = {}
    invalid: list[str] = []
    for key, value in dict(prompts or {}).items():
        if not isinstance(key, str) or not isinstance(value, str) or not _is_valid_prompt_key(key):
            invalid.append(str(key))
        else:
            valid[key] = value
    if invalid and strict:
        raise ValueError(f"invalid MTEB prompt keys in {source}: {', '.join(sorted(invalid))}")
    for key in sorted(invalid):
        logger.warning(
            "Ignoring unsupported MTEB prompt key %r from %s; it is not translated",
            key,
            source,
        )
    return valid


def load_model_overrides(root: Path | None = None) -> dict[tuple[str, str], ModelOverride]:
    """Load exact model behavior overrides from packaged or checkout YAML."""
    directory = _registry_directory(root)
    if not directory.is_dir():
        return {}
    overrides: dict[tuple[str, str], ModelOverride] = {}
    for path in sorted((*directory.glob("*.yaml"), *directory.glob("*.yml"))):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            try:
                override = ModelOverride.model_validate(entry)
                _filter_prompts(override.prompts, source=str(path), strict=True)
            except Exception as exc:
                raise ValueError(f"invalid model override in {path}: {exc}") from exc
            key = (override.hf_id, override.revision)
            if key in overrides:
                raise ValueError(
                    f"duplicate model override for {override.hf_id}@{override.revision}"
                )
            overrides[key] = override
    return overrides


def _validate_revision(revision: str) -> None:
    if len(revision) != SHA_LENGTH or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("model revision must be a full lowercase 40-character commit SHA")


def fingerprint_local_model(path: Path) -> tuple[str, int, int, float]:
    """Hash a local model tree using relative paths and complete file contents."""
    started = time.monotonic()
    digest = hashlib.sha256()
    files = sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    )
    total_bytes = 0
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        size = candidate.stat().st_size
        total_bytes += size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files), total_bytes, time.monotonic() - started


def _load_sentence_transformer(
    model_name: str,
    revision: str | None = None,
    device: str | None = None,
    model_prompts: dict[str, str] | None = None,
    **kwargs: Any,
):
    """Load a Hub SentenceTransformer while retaining only valid MTEB prompt keys."""
    from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name, revision=revision, device=device, **kwargs)
    native_prompts = _filter_prompts(
        getattr(encoder, "prompts", None), source=f"native prompts for {model_name}"
    )
    logger.info("Native SentenceTransformer prompts: %s", native_prompts)
    effective_prompts = {**native_prompts, **dict(model_prompts or {})}
    logger.info("Prompts after metadata and CLI overrides: %s", effective_prompts)
    encoder.prompts = {}
    wrapped = SentenceTransformerEncoderWrapper(model=encoder)
    wrapped.model_prompts = effective_prompts
    encoder.prompts = effective_prompts
    return wrapped


def _load_local_sentence_transformer(
    model_name: str,
    revision: str | None = None,
    device: str | None = None,
    local_path: str | None = None,
    model_prompts: dict[str, str] | None = None,
    **kwargs: Any,
):
    if local_path is None:
        raise ValueError("local model metadata is missing local_path")
    return _load_sentence_transformer(
        local_path,
        revision=None,
        device=device,
        model_prompts=model_prompts,
        **kwargs,
    )


def _resolve_local_model(
    path: Path,
    revision: str | None,
    *,
    allow_remote_code: bool,
    query_prompt: str | None,
    document_prompt: str | None,
) -> ModelMeta:
    if revision is not None:
        raise ValueError("--revision is not accepted for a local model path")
    if allow_remote_code:
        raise PermissionError("--allow-remote-code is not allowed for local model paths")
    path = path.resolve()
    logger.info("Detected local SentenceTransformer directory: %s", path)
    logger.info("Fingerprinting local model artifacts")
    digest, file_count, total_bytes, duration = fingerprint_local_model(path)
    logger.info(
        "Local fingerprint: files=%d bytes=%d duration=%.2fs sha256=%s",
        file_count,
        total_bytes,
        duration,
        digest,
    )
    prompts: dict[str, str] = {}
    if query_prompt is not None:
        prompts[PromptType.query.value] = query_prompt
    if document_prompt is not None:
        prompts[PromptType.document.value] = document_prompt
    name = f"local/{path.name or 'model'}"
    return ModelMeta.create_empty(
        overwrites={
            "name": name,
            "revision": f"{LOCAL_REVISION_PREFIX}{digest}",
            "loader": _load_local_sentence_transformer,
            "loader_kwargs": {
                "local_path": str(path),
                "model_prompts": prompts,
            },
            "reference": None,
        }
    )


def resolve_model(
    model: str,
    revision: str | None = None,
    *,
    allow_remote_code: bool = False,
    query_prompt: str | None = None,
    document_prompt: str | None = None,
) -> ModelMeta:
    """Resolve a Hub model at an exact SHA or a fingerprinted local model directory."""
    local_path = Path(model).expanduser()
    if local_path.exists():
        if not local_path.is_dir():
            raise ValueError(f"local model path is not a directory: {local_path}")
        return _resolve_local_model(
            local_path,
            revision,
            allow_remote_code=allow_remote_code,
            query_prompt=query_prompt,
            document_prompt=document_prompt,
        )

    if revision is None:
        logger.info(
            "No model revision supplied; resolving current Hugging Face Hub HEAD for %s",
            model,
        )
        hub_meta = ModelMeta.from_hub(model)
        resolved_revision = hub_meta.revision
        if resolved_revision is None:
            raise ValueError(f"could not resolve the current Hub revision for {model}")
        _validate_revision(resolved_revision)
        logger.info("Resolved current Hub revision: %s@%s", model, resolved_revision)
        try:
            meta = mteb.get_model_meta(model, revision=resolved_revision)
            metadata_source = "MTEB registry"
        except (KeyError, ValueError):
            meta = hub_meta
            metadata_source = "Hugging Face Hub"
    else:
        _validate_revision(revision)
        resolved_revision = revision
        try:
            meta = mteb.get_model_meta(model, revision=resolved_revision)
            metadata_source = "MTEB registry"
        except (KeyError, ValueError):
            meta = mteb.get_model_meta(
                model,
                revision=resolved_revision,
                fetch_from_hf=True,
                fill_missing=False,
            )
            metadata_source = "Hugging Face Hub"
    logger.info("Loaded model metadata from %s", metadata_source)

    override = load_model_overrides().get((model, resolved_revision))
    if override is not None:
        logger.info(
            "Applying exact NEB YAML override from %s for %s@%s",
            _registry_directory(),
            model,
            resolved_revision,
        )
    if override is not None and override.trust_remote_code:
        if not allow_remote_code:
            raise PermissionError(
                "this exact model override requires remote code; "
                "pass --allow-remote-code explicitly"
            )
    elif allow_remote_code:
        raise PermissionError("remote code is not approved for this exact model revision")

    loader_kwargs = dict(meta.loader_kwargs)
    upstream_prompts = _filter_prompts(
        loader_kwargs.get("model_prompts"), source=f"MTEB metadata for {model}"
    )
    override_prompts = _filter_prompts(
        override.prompts if override else None,
        source=f"NEB override for {model}@{resolved_revision}",
    )
    logger.info("MTEB metadata prompts: %s", upstream_prompts)
    logger.info("NEB YAML override prompts: %s", override_prompts)
    effective_prompts = {**upstream_prompts, **override_prompts}
    cli_prompts: dict[str, str] = {}
    if query_prompt is not None:
        effective_prompts[PromptType.query.value] = query_prompt
        cli_prompts[PromptType.query.value] = query_prompt
    if document_prompt is not None:
        effective_prompts[PromptType.document.value] = document_prompt
        cli_prompts[PromptType.document.value] = document_prompt
    logger.info("CLI prompt overrides: %s", cli_prompts)
    loader_kwargs["model_prompts"] = effective_prompts
    if override is not None and override.trust_remote_code:
        loader_kwargs["trust_remote_code"] = True

    updates: dict[str, Any] = {
        "name": model,
        "revision": resolved_revision,
        "loader_kwargs": loader_kwargs,
    }
    from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

    if meta.loader is SentenceTransformerEncoderWrapper:
        updates["loader"] = _load_sentence_transformer
    logger.info("Configured model prompts before native model load: %s", effective_prompts)
    return meta.model_copy(update=updates, deep=True)
