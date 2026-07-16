import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from mteb.models import ModelMeta
from mteb.models.abs_encoder import get_prompt_name
from mteb.types import PromptType

from neb.models import (
    _filter_prompts,
    fingerprint_local_model,
    load_model_overrides,
    resolve_model,
)
from neb.tasks import NanoBEIRNepaliRetrievalV3, STSBNepaliV3


def empty_meta(name: str, revision: str) -> ModelMeta:
    return ModelMeta.create_empty(overwrites={"name": name, "revision": revision})


def test_model_revision_must_be_full_sha() -> None:
    with pytest.raises(ValueError, match="40-character"):
        resolve_model("owner/model", "main")


def test_unregistered_public_model_uses_upstream_hub_resolution() -> None:
    meta = empty_meta("owner/model", "a" * 40)
    with patch("neb.models.mteb.get_model_meta", side_effect=[KeyError(), meta]) as get:
        result = resolve_model("owner/model", "a" * 40, query_prompt="search: ")
    assert get.call_count == 2
    assert get.call_args.kwargs["fetch_from_hf"] is True
    assert result.loader_kwargs["model_prompts"]["query"] == "search: "


def test_omitted_revision_resolves_current_hub_sha() -> None:
    revision = "b" * 40
    hub_meta = empty_meta("owner/model", revision)
    with (
        patch.object(ModelMeta, "from_hub", return_value=hub_meta) as from_hub,
        patch("neb.models.mteb.get_model_meta", return_value=hub_meta) as get,
    ):
        result = resolve_model("owner/model")
    from_hub.assert_called_once_with("owner/model")
    get.assert_called_once_with("owner/model", revision=revision)
    assert result.revision == revision


def test_passage_prompt_is_ignored_and_never_translated(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        prompts = _filter_prompts(
            {"query": "q: ", "document": "d: ", "passage": "wrong: "},
            source="fixture",
        )
    assert prompts == {"query": "q: ", "document": "d: "}
    assert "not translated" in caplog.text


def test_custom_neb_task_prompt_keys_are_valid() -> None:
    assert _filter_prompts(
        {
            "STSBNepali.v3": "similarity: ",
            "NanoBEIRNepaliRetrieval.v3-query": "search: ",
        },
        source="fixture",
        strict=True,
    ) == {
        "STSBNepali.v3": "similarity: ",
        "NanoBEIRNepaliRetrieval.v3-query": "search: ",
    }


def test_exact_yaml_override_uses_task_types_for_symmetric_e5_tasks() -> None:
    revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    override = load_model_overrides()[("intfloat/multilingual-e5-small", revision)]
    assert override.prompts["document"] == "passage: "
    assert get_prompt_name(override.prompts, STSBNepaliV3.metadata, None) == "STS"
    assert (
        get_prompt_name(
            override.prompts,
            NanoBEIRNepaliRetrievalV3.metadata,
            PromptType.document,
        )
        == "document"
    )


def test_model_overrides_reject_duplicate_exact_revisions(tmp_path: Path) -> None:
    directory = tmp_path / "registries" / "models"
    directory.mkdir(parents=True)
    entry = "- hf_id: owner/model\n  revision: " + "a" * 40 + "\n"
    (directory / "one.yaml").write_text(entry, encoding="utf-8")
    (directory / "two.yaml").write_text(entry, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate model override"):
        load_model_overrides(tmp_path)


def test_remote_code_requires_exact_yaml_override_and_flag() -> None:
    name = "jangedoo/embeddinggemma-300m-ne-pruned"
    revision = "4cc427f24b121ad66695ef2b45427ee6ea166907"
    with patch("neb.models.mteb.get_model_meta", return_value=empty_meta(name, revision)):
        with pytest.raises(PermissionError, match="allow-remote-code"):
            resolve_model(name, revision)
        approved = resolve_model(name, revision, allow_remote_code=True)
        assert approved.loader_kwargs["trust_remote_code"] is True
    with (
        patch(
            "neb.models.mteb.get_model_meta",
            return_value=empty_meta("jangedoo/other", "a" * 40),
        ),
        pytest.raises(PermissionError, match="not approved"),
    ):
        resolve_model("jangedoo/other", "a" * 40, allow_remote_code=True)


def test_local_model_fingerprint_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    model = tmp_path / "my-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    first = fingerprint_local_model(model)[0]
    assert fingerprint_local_model(model)[0] == first
    (model / "config.json").write_text('{"changed": true}', encoding="utf-8")
    assert fingerprint_local_model(model)[0] != first


def test_local_model_resolution_uses_unpublishable_fingerprint(tmp_path: Path) -> None:
    model = tmp_path / "my-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    meta = resolve_model(str(model), query_prompt="local query: ")
    assert meta.name == "local/my-model"
    assert meta.revision.startswith("local-")
    assert meta.loader_kwargs["local_path"] == str(model.resolve())
    assert meta.loader_kwargs["model_prompts"] == {"query": "local query: "}
    with pytest.raises(ValueError, match="not accepted"):
        resolve_model(str(model), "a" * 40)
    with pytest.raises(PermissionError, match="not allowed"):
        resolve_model(str(model), allow_remote_code=True)
