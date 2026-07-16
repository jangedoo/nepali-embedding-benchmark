from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from neb.schemas import VerificationStatus
from neb.wizard import collect_run_config, launch_wizard


class FakeUI:
    def __init__(
        self,
        *,
        texts: Sequence[str] = (),
        selects: Sequence[Any] = (),
        checkboxes: Sequence[list[Any]] = (),
        confirms: Sequence[bool] = (),
    ) -> None:
        self.texts = list(texts)
        self.selects = list(selects)
        self.checkboxes = list(checkboxes)
        self.confirms = list(confirms)
        self.output: list[str] = []
        self.text_defaults: dict[str, str] = {}
        self.select_defaults: dict[str, Any] = {}

    def text(
        self,
        message: str,
        *,
        default: str = "",
        validate: Callable[[str], bool | str] | None = None,
        instruction: str | None = None,
    ) -> str:
        self.text_defaults[message] = default
        value = self.texts.pop(0)
        if validate is not None:
            assert validate(value) is True
        return value

    def select(
        self,
        message: str,
        *,
        choices: Sequence[Any],
        default: Any = None,
        instruction: str | None = None,
    ) -> Any:
        self.select_defaults[message] = default
        return self.selects.pop(0)

    def checkbox(
        self,
        message: str,
        *,
        choices: Sequence[Any],
        validate: Callable[[list[Any]], bool | str] | None = None,
        instruction: str | None = None,
    ) -> list[Any]:
        value = self.checkboxes.pop(0)
        if validate is not None:
            assert validate(value) is True
        return value

    def confirm(self, message: str, *, default: bool = True) -> bool:
        return self.confirms.pop(0)

    def write(self, message: str = "", *, style: str | None = None) -> None:
        self.output.append(message)

    def assert_consumed(self) -> None:
        assert not self.texts
        assert not self.selects
        assert not self.checkboxes
        assert not self.confirms


def _task(name: str = "STSBNepali.v3") -> Any:
    metadata = SimpleNamespace(
        name=name,
        type="STS",
        dataset={"path": "jangedoo/stsb_nepali"},
    )
    return SimpleNamespace(metadata=metadata, hf_subsets=["ne-ne", "en-ne", "ne-en"])


def test_collect_run_config_covers_task_and_runtime_options() -> None:
    ui = FakeUI(
        texts=[
            "owner/model",
            "a" * 40,
            "64",
            "custom-runs",
            "query: ",
            "passage: ",
        ],
        selects=["choose", "cuda", "bf16", "DEBUG"],
        checkboxes=[["STSBNepali.v3"]],
        confirms=[True, True],
    )

    config = collect_run_config(ui, [_task()])

    assert config.model == "owner/model"
    assert config.revision == "a" * 40
    assert config.tasks == ["STSBNepali.v3"]
    assert config.device == "cuda"
    assert config.batch_size == 64
    assert config.dtype == "bf16"
    assert config.cache == Path("custom-runs")
    assert config.query_prompt == "query:"
    assert config.document_prompt == "passage:"
    assert config.allow_remote_code is True
    assert config.log_level == "DEBUG"
    assert ui.text_defaults["Batch size"] == "64"
    assert ui.select_defaults["Execution device"] == "cuda"
    assert ui.select_defaults["Model precision"] == "bf16"
    ui.assert_consumed()


def test_launch_wizard_runs_then_publishes_only_its_revision_and_exports(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "src/neb").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    cache = tmp_path / "runs"
    ui = FakeUI(
        texts=["owner/model", "", "64", str(cache)],
        selects=["choose", "cuda", "bf16", "community"],
        checkboxes=[["STSBNepali.v3"]],
        confirms=[False, True, False, True],
    )
    result_value = SimpleNamespace(
        model_name="owner/model",
        model_revision="b" * 40,
        task_names=["STSBNepali.v3"],
    )

    with (
        patch("neb.tasks.get_tasks", return_value=[_task()]),
        patch("neb.results.publish_results", return_value=[Path("published.json")]) as publish,
        patch("neb.export.export_static", return_value=[Path("catalog.json")]) as export,
    ):
        run = Mock(return_value=result_value)
        result = launch_wizard(root, run_evaluation=run, ui=ui)

    assert result is result_value
    assert run.call_args.args == (
        "owner/model",
        None,
        ["STSBNepali.v3"],
        "cuda",
        64,
        "bf16",
        False,
        None,
        None,
        cache,
        "INFO",
    )
    assert run.call_args.kwargs == {"json_summary": False}
    expected_source = cache.resolve() / "results/owner__model" / ("b" * 40)
    assert publish.call_args.args == (expected_source, VerificationStatus.community, root)
    assert publish.call_args.kwargs == {"overwrite": False}
    export.assert_called_once_with(root)
    assert any("No overall score" in line for line in ui.output)
    ui.assert_consumed()


def test_launch_wizard_does_not_offer_publication_for_local_results(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    ui = FakeUI(
        texts=[str(model), "16", "runs"],
        selects=["all", "cpu", None],
        confirms=[False, True],
    )
    result_value = SimpleNamespace(
        model_name="local/model",
        model_revision="local-abc",
        task_names=["STSBNepali.v3"],
    )

    with patch("neb.tasks.get_tasks", return_value=[_task()]):
        run = Mock(return_value=result_value)
        launch_wizard(tmp_path, run_evaluation=run, ui=ui)

    assert any("cannot be published" in line for line in ui.output)
    ui.assert_consumed()
