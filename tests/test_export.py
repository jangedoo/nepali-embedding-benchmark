import csv
import json
from pathlib import Path

import jsonschema
import yaml
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


def test_two_model_revisions_publish_and_export_separately(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    make_root(root)
    registry = root / "registries/models/jangedoo.yaml"
    entries = yaml.safe_load(registry.read_text(encoding="utf-8"))
    original = next(entry for entry in entries if entry["id"] == "all-minilm-l6-v2-nepali")
    revision = {
        **original,
        "id": "all-minilm-l6-v2-nepali-bbbbbbbb",
        "display_name": "all-MiniLM-L6-v2-nepali (rev bbbbbbbb)",
        "revision": "b" * 40,
    }
    entries.append(revision)
    registry.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")

    first = make_run(root, directory="first")
    second = make_run(
        root,
        model_id="all-minilm-l6-v2-nepali-bbbbbbbb",
        directory="second",
    )
    first_destination = publish_run(first, VerificationStatus.community, root)
    second_destination = publish_run(second, VerificationStatus.community, root)
    assert first_destination != second_destination

    output = tmp_path / "export"
    export_static(root, output)
    models = json.loads((output / "models.json").read_text(encoding="utf-8"))["models"]
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))["results"]
    expected_ids = {
        "all-minilm-l6-v2-nepali",
        "all-minilm-l6-v2-nepali-bbbbbbbb",
    }
    assert expected_ids <= {model["id"] for model in models}
    assert expected_ids == {result["model_id"] for result in results}
