"""Safe model-manifest scaffolding."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from neb.schemas import ModelSpec


def scaffold_model(hf_id: str, root: Path) -> Path:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to scaffold a model") from exc
    info = HfApi().model_info(hf_id)
    if info.private or info.gated:
        raise ValueError("community models must be public and ungated")
    if info.library_name != "sentence-transformers":
        raise ValueError("model must declare the sentence-transformers library")
    if "custom_code" in (info.tags or []):
        raise ValueError("community models must not require remote code")
    slug = re.sub(r"[^a-z0-9.-]+", "-", hf_id.lower().replace("/", "--")).strip("-")
    card_data = info.card_data or {}
    card = card_data.to_dict() if hasattr(card_data, "to_dict") else card_data
    spec = ModelSpec(
        id=slug,
        display_name=hf_id.split("/", 1)[1],
        hf_id=hf_id,
        revision=info.sha,
        license=str(card.get("license", "unknown")),
        homepage=f"https://huggingface.co/{hf_id}",
    )
    path = root / "registries/models" / f"{slug}.yaml"
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}")
    path.write_text(
        yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True), sort_keys=False),
        encoding="utf-8",
    )
    return path
