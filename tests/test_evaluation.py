from pathlib import Path

import pytest

from neb.evaluation import EvaluationRunner
from neb.schemas import ModelSpec, RuntimeSettings

ROOT = Path(__file__).parents[1]


class FakeEncoder:
    prompts = {"query": "native query: ", "document": "native document: "}


def test_manifest_prompt_overrides_native_prompt() -> None:
    spec = ModelSpec(
        id="model",
        display_name="Model",
        hf_id="owner/model",
        revision="a" * 40,
        prompts={"query": "override: "},
    )
    prompts = EvaluationRunner._effective_prompts(spec, FakeEncoder())
    assert prompts.query == "override: "
    assert prompts.document == "native document: "


def test_remote_code_requires_explicit_runtime_flag(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="allow-remote-code"):
        EvaluationRunner(ROOT).run(
            "multilingual-e5-small-pruned",
            ["stsb-nepali"],
            runtime=RuntimeSettings(),
            allow_remote_code=False,
            output_dir=tmp_path,
        )
