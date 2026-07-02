import csv
import json
from pathlib import Path

import jsonschema

from neb.export import export_static

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
    schema = json.loads((ROOT / "schemas/export-v1.schema.json").read_text())
    for name in ("catalog.json", "models.json", "tasks.json", "results.json"):
        jsonschema.validate(json.loads((tmp_path / name).read_text()), schema)
    with (tmp_path / "results.csv").open(newline="") as stream:
        assert csv.DictReader(stream).fieldnames is not None
