import json
from pathlib import Path

import pytest
from conftest import make_sts_cache

from neb.results import discover_results, publish_results, validate_task_result
from neb.schemas import VerificationStatus


def test_native_result_validation_and_checksum(tmp_path: Path) -> None:
    path = make_sts_cache(tmp_path / "runs")
    _, _, records = validate_task_result(path)
    assert records[0].main_score_name == "cosine_spearman"
    assert records[0].effective_prompts == {"query": "query: "}
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="TaskResult|checksum"):
        validate_task_result(path)


def test_publication_scans_a_native_cache_without_reading_model_metadata_as_a_task(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "runs"
    make_sts_cache(cache)
    published = publish_results(cache, VerificationStatus.community, tmp_path / "repo")
    assert [path.name for path in published] == ["STSBNepali.v3.json"]


def test_validation_rejects_missing_prompts_and_settings(tmp_path: Path) -> None:
    path = make_sts_cache(tmp_path / "runs")
    meta_path = path.parent / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["loader_kwargs"].pop("model_prompts")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="effective prompts"):
        validate_task_result(path)


def test_publication_rejects_local_model_results(tmp_path: Path) -> None:
    path = make_sts_cache(tmp_path / "runs")
    meta_path = path.parent / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["name"] = "local/my-model"
    meta["revision"] = "local-" + "a" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="local model results cannot be published"):
        publish_results(path, VerificationStatus.community, tmp_path / "repo")


def test_publication_rejects_conflicts_but_allows_partial_additions(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    first = make_sts_cache(tmp_path / "first", score=0.5, subset="ne-ne")
    publish_results(first, VerificationStatus.community, root)
    second = make_sts_cache(tmp_path / "second", score=0.6, subset="en-ne")
    publish_results(second, VerificationStatus.community, root)
    assert {record.subset for record in discover_results(root)} == {"ne-ne", "en-ne"}
    conflict = make_sts_cache(tmp_path / "conflict", score=0.9, subset="ne-ne")
    with pytest.raises(ValueError, match="conflicting"):
        publish_results(conflict, VerificationStatus.community, root)


def test_verified_precedence_is_per_subset(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    community = make_sts_cache(tmp_path / "community", score=0.2, subset="ne-ne")
    verified = make_sts_cache(tmp_path / "verified", score=0.8, subset="ne-ne")
    other = make_sts_cache(tmp_path / "other", score=0.4, subset="en-ne")
    publish_results(community, VerificationStatus.community, root)
    publish_results(other, VerificationStatus.community, root)
    publish_results(verified, VerificationStatus.verified, root)
    records = discover_results(root)
    selected = {record.subset: record for record in records}
    assert selected["ne-ne"].status == VerificationStatus.verified
    assert selected["ne-ne"].main_score == 0.8
    assert selected["en-ne"].status == VerificationStatus.community
