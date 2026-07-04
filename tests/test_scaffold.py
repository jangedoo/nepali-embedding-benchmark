from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from neb import scaffold
from neb.registry import load_models


def info(revision: str, **overrides: object) -> SimpleNamespace:
    values = {
        "sha": revision,
        "private": False,
        "gated": False,
        "library_name": "sentence-transformers",
        "pipeline_tag": None,
        "tags": [],
        "card_data": {"license": "apache-2.0"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def write_models(root: Path, filename: str, entries: object) -> Path:
    path = root / "registries/models" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
    return path


def model(model_id: str, hf_id: str, revision: str) -> dict[str, object]:
    return {
        "id": model_id,
        "display_name": hf_id.split("/")[1],
        "hf_id": hf_id,
        "revision": revision,
    }


def test_dedicated_owner_creates_lowercase_list_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scaffold, "_fetch_model_info", lambda _: info("a" * 40))

    result = scaffold.scaffold_model("Alibaba-NLP/example", tmp_path)

    assert result.action == "created"
    assert result.model_id == "alibaba-nlp--example"
    assert result.path == tmp_path / "registries/models/alibaba-nlp.yaml"
    raw = yaml.safe_load(result.path.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert raw[0]["hf_id"] == "Alibaba-NLP/example"


def test_unknown_owner_appends_to_community_and_converts_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_models(tmp_path, "community.yaml", model("old", "other/old", "a" * 40))
    monkeypatch.setattr(scaffold, "_fetch_model_info", lambda _: info("b" * 40))

    scaffold.scaffold_model("new-owner/new-model", tmp_path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in raw] == ["old", "new-owner--new-model"]


@pytest.mark.parametrize("pipeline_tag", ["feature-extraction", "fill-mask"])
def test_transformers_text_encoder_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipeline_tag: str
) -> None:
    monkeypatch.setattr(
        scaffold,
        "_fetch_model_info",
        lambda _: info(
            "a" * 40,
            library_name="transformers",
            pipeline_tag=pipeline_tag,
        ),
    )

    result = scaffold.scaffold_model("owner/model", tmp_path)

    assert result.action == "created"
    assert result.model_id == "owner--model"


def test_exact_revision_is_noop_across_registry_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_models(tmp_path, "legacy.yaml", [model("stable-id", "owner/model", "a" * 40)])
    before = path.read_bytes()
    monkeypatch.setattr(scaffold, "_fetch_model_info", lambda _: info("a" * 40))

    result = scaffold.scaffold_model("owner/model", tmp_path)

    assert result.action == "existing"
    assert result.model_id == "stable-id"
    assert result.path == path
    assert path.read_bytes() == before
    assert not (tmp_path / "registries/models/community.yaml").exists()


def test_new_revision_preserves_old_entry_and_extends_colliding_sha_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = model("qwen--model", "Qwen/model", "a" * 40)
    collision = model("qwen--model-bbbbbbbb", "other/collision", "c" * 40)
    path = write_models(tmp_path, "qwen.yaml", old)
    write_models(tmp_path, "community.yaml", [collision])
    monkeypatch.setattr(scaffold, "_fetch_model_info", lambda _: info("b" * 40))

    result = scaffold.scaffold_model("Qwen/model", tmp_path)

    assert result.action == "revision"
    assert result.model_id == "qwen--model-bbbbbbbbb"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in raw] == ["qwen--model", "qwen--model-bbbbbbbbb"]
    assert raw[1]["display_name"] == "model (rev bbbbbbbb)"
    assert len(load_models(tmp_path)) == 3


@pytest.mark.parametrize("contents", ["", "- broken: [yaml", "42\n", "[]\n"])
def test_malformed_or_empty_registry_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str
) -> None:
    path = tmp_path / "registries/models/community.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(scaffold, "_fetch_model_info", lambda _: info("a" * 40))

    with pytest.raises(ValueError, match="registry|YAML"):
        scaffold.scaffold_model("owner/model", tmp_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"private": True}, "public and ungated"),
        ({"gated": True}, "public and ungated"),
        ({"library_name": "pytorch"}, "sentence-transformers or transformers"),
        ({"tags": ["custom_code"]}, "remote code"),
    ],
)
def test_model_policy_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(scaffold, "_fetch_model_info", lambda _: info("a" * 40, **overrides))

    with pytest.raises(ValueError, match=message):
        scaffold.scaffold_model("owner/model", tmp_path)


@pytest.mark.parametrize(
    "pipeline_tag",
    [
        None,
        "text-generation",
        "text-classification",
        "image-classification",
        "automatic-speech-recognition",
        "image-text-to-text",
    ],
)
def test_transformers_unsupported_pipeline_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_tag: str | None,
) -> None:
    monkeypatch.setattr(
        scaffold,
        "_fetch_model_info",
        lambda _: info(
            "a" * 40,
            library_name="transformers",
            pipeline_tag=pipeline_tag,
        ),
    )

    with pytest.raises(ValueError, match="feature-extraction or fill-mask"):
        scaffold.scaffold_model("owner/model", tmp_path)


def test_model_revision_must_be_a_full_commit_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scaffold, "_fetch_model_info", lambda _: info("abc123"))

    with pytest.raises(ValueError, match="full 40-character"):
        scaffold.scaffold_model("owner/model", tmp_path)
