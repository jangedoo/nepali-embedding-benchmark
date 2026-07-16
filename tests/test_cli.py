import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from neb.cli import app
from neb.schemas import VerificationStatus


def test_run_allows_omitted_revision_and_keeps_summary_on_stdout() -> None:
    result_value = SimpleNamespace(
        model_name="owner/model",
        model_revision="a" * 40,
        task_names=["STSBNepali.v3"],
    )
    with patch("neb.evaluation.evaluate", return_value=result_value) as evaluate:
        result = CliRunner().invoke(
            app,
            ["run", "--model", "owner/model", "--task", "STSBNepali.v3"],
        )
    assert result.exit_code == 0
    assert evaluate.call_args.args[1] is None
    assert json.loads(result.stdout)["model_revision"] == "a" * 40


def test_run_rejects_invalid_log_level_before_evaluation() -> None:
    with patch("neb.evaluation.evaluate") as evaluate:
        result = CliRunner().invoke(
            app,
            ["run", "--model", "owner/model", "--log-level", "LOUD"],
        )
    assert result.exit_code != 0
    assert not evaluate.called


def test_results_publish_forwards_overwrite() -> None:
    with patch("neb.results.publish_results", return_value=[]) as publish:
        result = CliRunner().invoke(
            app,
            ["results", "publish", "runs", "--status", "verified", "--overwrite"],
        )
    assert result.exit_code == 0
    assert publish.call_args.args[1] == VerificationStatus.verified
    assert publish.call_args.kwargs == {"overwrite": True}
    assert publish.call_args.args[0] == Path("runs").resolve()
