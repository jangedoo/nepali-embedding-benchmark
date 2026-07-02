from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from neb.evaluation import EvaluationRunner, sha256_file
from neb.registry import load_models, load_tasks
from neb.results import discover_results, publish_run, validate_run
from neb.schemas import RunProvenance, RuntimeSettings, VerificationStatus

SOURCE_ROOT = Path(__file__).parents[1]


def make_root(tmp_path: Path) -> Path:
    shutil.copytree(SOURCE_ROOT / "registries", tmp_path / "registries")
    return tmp_path


def make_run(root: Path) -> Path:
    model = next(item for item in load_models(root) if item.id == "all-minilm-l6-v2-nepali")
    task = next(item for item in load_tasks(root) if item.id == "stsb-nepali")
    run = root / "incoming"
    result_dir = run / "results"
    result_dir.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for index, view in enumerate(task.views):
        path = result_dir / f"{view.id}.json"
        payload = EvaluationRunner._mteb_result(model, task, view, 0.5 + index / 10)
        path.write_text(json.dumps(payload), encoding="utf-8")
        hashes[f"results/{view.id}.json"] = sha256_file(path)
    provenance = RunProvenance(
        run_id="test-run",
        status="community",
        model_id=model.id,
        model_hf_id=model.hf_id,
        model_revision=model.revision,
        task_id=task.id,
        task_version=task.version,
        dataset_revision=task.dataset.revision,
        neb_version="0.1.0",
        sentence_transformers_version="4.0.0",
        runtime=RuntimeSettings(),
        result_hashes=hashes,
    )
    (run / "provenance.json").write_text(provenance.model_dump_json(), encoding="utf-8")
    (run / "model_meta.json").write_text(model.model_dump_json(), encoding="utf-8")
    (run / "run_settings.jsonl").write_text(RuntimeSettings().model_dump_json() + "\n")
    return run


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    run = make_run(root)
    (run / "results/ne-ne.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_run(run, root)


def test_verified_result_takes_precedence(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    run = make_run(root)
    publish_run(run, VerificationStatus.community, root)
    publish_run(run, VerificationStatus.verified, root)
    records = discover_results(root)
    assert len(records) == 3
    assert {record.status for record in records} == {VerificationStatus.verified}
    with pytest.raises(FileExistsError):
        publish_run(run, VerificationStatus.community, root)
    existing = publish_run(
        run,
        VerificationStatus.community,
        root,
        skip_existing=True,
    )
    assert existing.is_dir()
