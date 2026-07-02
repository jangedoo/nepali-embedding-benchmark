import csv
import json
from pathlib import Path

import jsonschema
from test_results import make_root, make_run

from neb.export import export_static
from neb.results import publish_run
from neb.schemas import VerificationStatus

ROOT = Path(__file__).parents[1]


def test_static_export_contract(tmp_path: Path) -> None:
    paths = export_static(ROOT, tmp_path)
    assert {path.name for path in paths} == {
        "catalog.json",
        "models.json",
        "tasks.json",
        "results.json",
        "results.csv",
    }
    schema = json.loads((ROOT / "schemas/export-v2.schema.json").read_text())
    for name in ("catalog.json", "models.json", "tasks.json", "results.json"):
        jsonschema.validate(json.loads((tmp_path / name).read_text()), schema)
    models = {
        model["id"]: model for model in json.loads((tmp_path / "models.json").read_text())["models"]
    }
    assert isinstance(models["multilingual-e5-small"]["parameter_count"], int)
    assert isinstance(models["multilingual-e5-small"]["vocab_size"], int)
    assert models["qwen--qwen3-embedding-0.6b"]["parameter_count"] == "unknown"
    assert models["qwen--qwen3-embedding-0.6b"]["vocab_size"] == "unknown"
    with (tmp_path / "results.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        assert {"metric", "score", "is_primary"} <= set(reader.fieldnames)


def test_results_json_is_one_record_per_view_and_csv_is_long_form(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    make_root(root)
    publish_run(make_run(root), VerificationStatus.community, root)
    output = tmp_path / "export"
    export_static(root, output)

    results = json.loads((output / "results.json").read_text())["results"]
    assert len(results) == 3
    assert all(
        set(result["metrics"]) == {"cosine_spearman", "cosine_pearson"} for result in results
    )
    with (output / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert sum(row["is_primary"] == "True" for row in rows) == 3
    assert [row["metric"] for row in rows[:2]] == ["cosine_spearman", "cosine_pearson"]
