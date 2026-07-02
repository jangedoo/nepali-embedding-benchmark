"""Safe model-manifest scaffolding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neb.schemas import ModelSpec

DEDICATED_OWNERS = {
    "google",
    "qwen",
    "facebook",
    "alibaba-nlp",
    "microsoft",
    "baai",
    "nvidia",
    "intfloat",
    "sentence-transformers",
    "jinaai",
    "nomic-ai",
    "snowflake",
    "mixedbread-ai",
}


@dataclass(frozen=True)
class ScaffoldResult:
    action: str
    model_id: str
    path: Path

    def __str__(self) -> str:
        messages = {
            "created": "Created model",
            "revision": "Added revision as model",
            "existing": "Found existing model",
        }
        return f"{messages[self.action]} {self.model_id!r} in {self.path}"


def _fetch_model_info(hf_id: str) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to scaffold a model") from exc
    return HfApi().model_info(hf_id)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", value.lower()).strip("-")


def _registry_path(hf_id: str, root: Path) -> Path:
    owner = hf_id.split("/", 1)[0]
    owner_slug = _slug(owner)
    filename = f"{owner_slug}.yaml" if owner.lower() in DEDICATED_OWNERS else "community.yaml"
    return root / "registries/models" / filename


def _read_registry(path: Path) -> tuple[Any, list[ModelSpec]]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid model registry YAML in {path}: {exc}") from exc
    if raw is None or not isinstance(raw, (dict, list)):
        raise ValueError(f"model registry must contain a mapping or non-empty list: {path}")
    values = raw if isinstance(raw, list) else [raw]
    if not values:
        raise ValueError(f"model registry must not be empty: {path}")
    specs: list[ModelSpec] = []
    for value in values:
        try:
            specs.append(ModelSpec.model_validate(value))
        except Exception as exc:
            raise ValueError(f"invalid model registry entry in {path}: {exc}") from exc
    return raw, specs


def _all_registrations(root: Path) -> list[tuple[Path, ModelSpec]]:
    directory = root / "registries/models"
    registrations: list[tuple[Path, ModelSpec]] = []
    seen_ids: set[str] = set()
    for path in sorted((*directory.glob("*.yaml"), *directory.glob("*.yml"))):
        _, specs = _read_registry(path)
        for spec in specs:
            if spec.id in seen_ids:
                raise ValueError(f"duplicate model id: {spec.id}")
            seen_ids.add(spec.id)
            registrations.append((path, spec))
    return registrations


def _revision_id(base_id: str, revision: str, used_ids: set[str]) -> str:
    for length in range(8, len(revision) + 1):
        candidate = f"{base_id}-{revision[:length]}"
        if candidate not in used_ids:
            return candidate
    raise ValueError(f"cannot generate a unique model id for revision {revision}")


def scaffold_model(hf_id: str, root: Path) -> ScaffoldResult:
    if hf_id.count("/") != 1 or any(not part for part in hf_id.split("/")):
        raise ValueError("Hugging Face model id must have the form owner/model")

    info = _fetch_model_info(hf_id)
    if info.private or info.gated:
        raise ValueError("community models must be public and ungated")
    if info.library_name != "sentence-transformers":
        raise ValueError("model must declare the sentence-transformers library")
    if "custom_code" in (info.tags or []):
        raise ValueError("community models must not require remote code")
    if not isinstance(info.sha, str) or re.fullmatch(r"[0-9a-f]{40}", info.sha) is None:
        raise ValueError("model revision must be a full 40-character commit SHA")

    registrations = _all_registrations(root)
    same_model = [(path, spec) for path, spec in registrations if spec.hf_id == hf_id]
    for path, spec in same_model:
        if spec.revision == info.sha:
            return ScaffoldResult("existing", spec.id, path)

    base_id = same_model[0][1].id if same_model else _slug(hf_id.replace("/", "--"))
    used_ids = {spec.id for _, spec in registrations}
    is_revision = bool(same_model)
    if not is_revision and base_id in used_ids:
        raise ValueError(f"generated model id is already registered: {base_id}")
    model_id = _revision_id(base_id, info.sha, used_ids) if is_revision else base_id
    revision_label = f" (rev {info.sha[:8]})" if is_revision else ""
    card_data = info.card_data or {}
    card = card_data.to_dict() if hasattr(card_data, "to_dict") else card_data
    spec = ModelSpec(
        id=model_id,
        display_name=f"{hf_id.split('/', 1)[1]}{revision_label}",
        hf_id=hf_id,
        revision=info.sha,
        license=str(card.get("license", "unknown")),
        homepage=f"https://huggingface.co/{hf_id}",
    )

    path = _registry_path(hf_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[Any] = []
    if path.exists():
        raw, _ = _read_registry(path)
        existing = raw if isinstance(raw, list) else [raw]
    existing.append(spec.model_dump(mode="json", exclude_none=True))
    path.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")
    return ScaffoldResult("revision" if is_revision else "created", model_id, path)
