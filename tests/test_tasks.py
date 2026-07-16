from unittest.mock import patch

from mteb.abstasks import AbsTaskRetrieval

from neb.tasks import (
    RETRIEVAL_K_VALUES,
    NanoBEIRNepaliRetrievalV5,
    get_benchmark,
    get_result_tasks,
    get_tasks,
)


def test_benchmark_composition_has_no_aggregation() -> None:
    benchmark = get_benchmark()
    assert benchmark.name == "NEB(Nepali, v1)"
    assert benchmark.aggregations == []
    assert len(benchmark.tasks) == 10


def test_custom_versions_and_dataset_revisions_are_exact() -> None:
    tasks = get_tasks()
    assert [task.metadata.name for task in tasks[:7]] == [
        "STSBNepali.v3",
        "NanoBEIRNepaliRetrieval.v5",
        "NepaliHardNegativesRetrieval.v5",
        "NepaliEcommerceRetrieval.v2",
        "SanoIRGeneralRetrieval.v2",
        "NepaliParaphraseClassification.v3",
        "EnglishNepaliBitextMining.v3",
    ]
    assert all(len(task.metadata.revision) == 40 for task in tasks)
    assert all(set(task.metadata.modalities) == {"text"} for task in tasks)


def test_builtin_tasks_have_fixed_nepali_subsets() -> None:
    by_name = {task.metadata.name: task for task in get_tasks()}
    assert by_name["NepaliNewsClassification.v2"].hf_subsets == ["default"]
    assert by_name["IndicGenBenchFloresBitextMining"].hf_subsets == [
        "nep-eng",
        "eng-nep",
    ]
    assert set(by_name["NTREXBitextMining"].hf_subsets) == {
        "nep_Deva-eng_Latn",
        "eng_Latn-nep_Deva",
    }


def test_iso_language_script_codes_are_recorded() -> None:
    for task in get_tasks():
        mapping = task.metadata.hf_subsets_to_langscripts
        assert mapping
        assert all("-" in language for values in mapping.values() for language in values)


def test_retrieval_tasks_only_return_selected_metric_families() -> None:
    raw = {
        "ndcg_at_10": 0.8,
        "map_at_10": 0.7,
        "mrr_at_10": 0.6,
        "recall_at_10": 0.5,
        "hit_rate_at_10": 0.4,
        "precision_at_10": 0.3,
        "accuracy": 0.2,
        "nauc_ndcg_at_10_max": 0.1,
    }
    with patch.object(AbsTaskRetrieval, "_evaluate_subset", return_value=raw):
        scores = NanoBEIRNepaliRetrievalV5()._evaluate_subset()
    assert set(scores) == {
        "ndcg_at_10",
        "map_at_10",
        "mrr_at_10",
        "recall_at_10",
        "hit_rate_at_10",
    }


def test_retrieval_tasks_use_only_selected_cutoffs() -> None:
    retrieval_tasks = [task for task in get_result_tasks() if task.metadata.type == "Retrieval"]
    assert retrieval_tasks
    assert all(task.k_values == RETRIEVAL_K_VALUES for task in retrieval_tasks)
    assert all(task._top_k == 50 for task in retrieval_tasks)


def test_legacy_protocol_validates_evidence_without_remaining_active() -> None:
    active = {task.metadata.name for task in get_tasks()}
    result_tasks = {task.metadata.name for task in get_result_tasks()}
    assert "NanoBEIRNepaliRetrieval.v3" not in active
    assert "NanoBEIRNepaliRetrieval.v3" in result_tasks
    assert all("Reranking" not in name for name in result_tasks)
