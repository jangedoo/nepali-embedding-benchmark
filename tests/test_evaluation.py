from pathlib import Path

import mteb
from datasets import Dataset, DatasetDict
from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.results import TaskResult
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import BoW

from neb.evaluation import evaluate


class TinySTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="TinyNEBSTS",
        dataset={"path": "fixture/tiny", "revision": "a" * 40},
        description="In-memory native MTEB fixture.",
        type="STS",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["nep-Deva"],
        main_score="cosine_spearman",
        domains=["Constructed"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
    )


def test_native_mteb_evaluate_produces_task_result_without_network(tmp_path: Path) -> None:
    task = TinySTS()
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sentence1": ["क", "ख", "ग"],
                    "sentence2": ["क", "ख फरक", "एकदम फरक"],
                    "score": [5.0, 3.0, 0.0],
                }
            )
        }
    )
    task.data_loaded = True
    model = mteb.get_model_meta("mteb/baseline-random-encoder").load_model(device="cpu")
    result = mteb.evaluate(
        model,
        [task],
        cache=mteb.ResultCache(tmp_path),
        co2_tracker=False,
        show_progress_bar=False,
        encode_kwargs={"batch_size": 2},
    )
    assert len(result.task_results) == 1
    assert isinstance(result.task_results[0], TaskResult)
    assert "cosine_spearman" in result.task_results[0].scores["test"][0]


def test_neb_evaluate_supports_fingerprinted_local_sentence_transformer(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "tiny-local"
    SentenceTransformer(
        modules=[BoW(vocab=["क", "ख", "ग", "फरक", "एकदम"])],
        prompts={"query": "", "document": ""},
    ).save_pretrained(str(model_path))

    task = TinySTS()
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sentence1": ["क", "ख", "ग"],
                    "sentence2": ["क", "ख फरक", "एकदम फरक"],
                    "score": [5.0, 3.0, 0.0],
                }
            )
        }
    )
    task.data_loaded = True
    cache_path = tmp_path / "cache"
    result = evaluate(
        str(model_path),
        tasks=[task],
        cache_path=cache_path,
        show_progress_bar=False,
        encode_kwargs={"show_progress_bar": False},
    )

    assert result.model_name == "local/tiny-local"
    assert result.model_revision.startswith("local-")
    result_path = mteb.ResultCache(cache_path).get_task_result_path(
        "TinyNEBSTS", result.model_name, result.model_revision
    )
    assert result_path.is_file()
    assert result_path.with_suffix(".json.sha256").is_file()
