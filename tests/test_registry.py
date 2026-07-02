from pathlib import Path

from neb.registry import validate_registries

ROOT = Path(__file__).parents[1]


def test_seed_registry() -> None:
    tasks, models = validate_registries(ROOT)
    assert len(tasks) == 5
    assert len(models) == 9
    assert sum(len(task.views) for task in tasks) == 20
    assert all(len(task.dataset.revision) == 40 for task in tasks)
    assert all(len(model.revision) == 40 for model in models)


def test_nanobeir_has_all_separate_views() -> None:
    tasks, _ = validate_registries(ROOT)
    task = next(item for item in tasks if item.id == "nanobeir-ne")
    assert len(task.views) == 13
    expected = {"corpus": "corpus", "queries": "queries", "qrels": "qrels"}
    assert all(view.resources == expected for view in task.views)
