from neb.tasks import get_benchmark, get_tasks


def test_benchmark_composition_has_no_aggregation() -> None:
    benchmark = get_benchmark()
    assert benchmark.name == "NEB(Nepali, v1)"
    assert benchmark.aggregations == []
    assert len(benchmark.tasks) == 8


def test_custom_versions_and_dataset_revisions_are_exact() -> None:
    tasks = get_tasks()
    custom = tasks[:5]
    assert all(task.metadata.name.endswith(".v3") for task in custom)
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
