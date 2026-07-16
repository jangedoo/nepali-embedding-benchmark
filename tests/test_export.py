import json
from pathlib import Path

from conftest import make_sts_cache
from jsonschema import validate

from neb.export import export_static
from neb.results import publish_results
from neb.schemas import VerificationStatus


def test_empty_v3_export_has_tasks_and_no_fabricated_scores(tmp_path: Path) -> None:
    paths = export_static(tmp_path, tmp_path / "data")
    catalog = json.loads((tmp_path / "data/catalog.json").read_text(encoding="utf-8"))
    assert catalog["schema_version"] == 3
    assert catalog["counts"] == {"tasks": 8, "models": 0, "results": 0}
    assert len(paths) == 5


def test_export_groups_model_revisions_and_marks_latest(tmp_path: Path) -> None:
    older = make_sts_cache(tmp_path / "older", revision="a" * 40)
    newer = make_sts_cache(tmp_path / "newer", revision="b" * 40)
    root = tmp_path / "repo"
    publish_results(older, VerificationStatus.verified, root)
    publish_results(newer, VerificationStatus.verified, root)
    export_static(root, root / "site/public/data/v3")
    catalog = json.loads((root / "site/public/data/v3/catalog.json").read_text(encoding="utf-8"))
    assert {model["repository"] for model in catalog["models"]} == {"owner/model"}
    assert sum(model["is_latest"] for model in catalog["models"]) == 1
    result = catalog["results"][0]
    assert result["dataset_revision"] and result["result_sha256"]
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas/export-v3.schema.json").read_text(encoding="utf-8")
    )
    validate(catalog, schema)
