from pathlib import Path

import pytest

from neb.evaluation import (
    EvaluationRunner,
    _lookup,
    bitext_metrics,
    pair_classification_metrics,
    reranking_metrics,
    resolve_model_dtype,
    retrieval_metrics,
    select_classification_threshold,
    sts_metrics,
)
from neb.schemas import ModelSpec, RuntimeSettings

ROOT = Path(__file__).parents[1]


class FakeEncoder:
    prompts = {"query": "native query: ", "document": "native document: "}


class FakeParameter:
    def __init__(self, size: int):
        self.size = size

    def numel(self) -> int:
        return self.size


class FakeModel(FakeEncoder):
    tokenizer = list(range(30))

    @staticmethod
    def parameters() -> list[FakeParameter]:
        return [FakeParameter(10), FakeParameter(20)]


class FakeCuda:
    def __init__(self, supports_bf16: bool):
        self.supports_bf16 = supports_bf16

    def is_bf16_supported(self) -> bool:
        return self.supports_bf16


class FakeTorch:
    bfloat16 = object()
    float16 = object()
    float32 = object()

    def __init__(self, supports_bf16: bool):
        self.cuda = FakeCuda(supports_bf16)


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


def test_model_stats_are_extracted_from_loaded_encoder() -> None:
    assert EvaluationRunner._model_stats(FakeModel()) == (30, 30)


@pytest.mark.parametrize(
    ("supports_bf16", "expected_name", "expected_dtype"),
    [(True, "bfloat16", FakeTorch.bfloat16), (False, "float16", FakeTorch.float16)],
)
def test_cuda_model_dtype_is_selected_from_hardware_support(
    supports_bf16: bool, expected_name: str, expected_dtype: object
) -> None:
    runtime, dtype = resolve_model_dtype(RuntimeSettings(device="cuda"), FakeTorch(supports_bf16))
    assert runtime.dtype == expected_name
    assert dtype is expected_dtype


def test_explicit_model_dtype_override_is_respected() -> None:
    runtime, dtype = resolve_model_dtype(
        RuntimeSettings(device="cuda", dtype="fp32"), FakeTorch(True)
    )
    assert runtime.dtype == "float32"
    assert dtype is FakeTorch.float32


def test_cpu_model_dtype_remains_unspecified_by_default() -> None:
    original = RuntimeSettings(device="cpu")
    runtime, dtype = resolve_model_dtype(original, FakeTorch(True))
    assert runtime is original
    assert dtype is None


def test_sts_metric_family() -> None:
    metrics = sts_metrics([0.1, 0.4, 0.8], [1.0, 2.0, 3.0])
    assert metrics["cosine_spearman"] == pytest.approx(1)
    assert 0.9 < metrics["cosine_pearson"] <= 1


def test_retrieval_metric_family() -> None:
    import numpy as np

    metrics = retrieval_metrics(
        np.array([[1, 0, 0], [0, 1, 0]]),
        np.array([[0.9, 0.2, 0.1], [0.8, 0.7, 0.1]]),
    )
    assert metrics["ndcg_at_10"] == pytest.approx((1 + 1 / 1.5849625) / 2, abs=1e-5)
    assert metrics["mrr_at_10"] == pytest.approx(0.75)
    assert metrics["hit_rate_at_10"] == 1


def test_reranking_and_bitext_metric_families() -> None:
    reranking = reranking_metrics([1, 2, 6])
    assert reranking["hit_rate_at_1"] == pytest.approx(1 / 3)
    assert reranking["hit_rate_at_3"] == pytest.approx(2 / 3)
    bitext = bitext_metrics([0, 2, 1], [0, 1, 2])
    assert bitext == {
        "f1": pytest.approx(1 / 3),
        "precision": pytest.approx(1 / 3),
        "recall": pytest.approx(1 / 3),
        "accuracy": pytest.approx(1 / 3),
    }


def test_paraphrase_threshold_is_selected_on_validation_scores() -> None:
    threshold = select_classification_threshold([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    assert 0.3 < threshold < 0.7
    metrics = pair_classification_metrics([0, 1, 1], [0.2, 0.6, 0.8], threshold)
    assert set(metrics) == {
        "cosine_ap",
        "cosine_roc_auc",
        "cosine_accuracy",
        "cosine_f1",
        "cosine_precision",
        "cosine_recall",
    }
    assert metrics["cosine_accuracy"] == 1


def test_remote_code_requires_explicit_runtime_flag(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="allow-remote-code"):
        EvaluationRunner(ROOT).run(
            "multilingual-e5-small-pruned",
            ["stsb-nepali"],
            runtime=RuntimeSettings(),
            allow_remote_code=False,
            output_dir=tmp_path,
        )


def test_model_lookup_accepts_unique_hf_id() -> None:
    spec = ModelSpec(
        id="model-a",
        display_name="Model",
        hf_id="owner/model",
        revision="a" * 40,
    )
    assert _lookup([spec], "owner/model", "model") is spec


def test_model_lookup_rejects_ambiguous_hf_id_with_valid_ids() -> None:
    specs = [
        ModelSpec(
            id=f"model-{suffix}",
            display_name="Model",
            hf_id="owner/model",
            revision=revision * 40,
        )
        for suffix, revision in [("a", "a"), ("b", "b")]
    ]
    with pytest.raises(ValueError, match=r"ambiguous.*model-a, model-b"):
        _lookup(specs, "owner/model", "model")

    assert _lookup(specs, "model-b", "model") is specs[1]
