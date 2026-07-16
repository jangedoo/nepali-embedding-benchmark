"""Native MTEB task definitions and NEB benchmark composition."""

from __future__ import annotations

from typing import Any

import mteb
from datasets import Dataset, DatasetDict, load_dataset
from mteb.abstasks import AbsTaskPairClassification, AbsTaskRetrieval, AbsTaskSTS
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

from neb.adapters import (
    normalize_hard_negative_retrieval_rows,
    normalize_hard_negative_rows,
    normalize_nanobeir,
    normalize_parallel_direction,
    normalize_row_retrieval,
    normalize_sts_rows,
)

STSB_REVISION = "4bfcc74faa37875185b27a5dc64888f4711e833b"
NANOBEIR_REVISION = "dbe0a6befdfce448982392bd461ac5d16485e99e"
HARD_NEGATIVES_REVISION = "f92818df24ed4d95905ad928dbde18feb4aeb2fb"
ECOMMERCE_REVISION = "4b7dd8188039018701baa94eb56a1a63d0ab16d5"
SANOIR_REVISION = "7e53250964a040dbad2aa11d95855548a7c6717e"
PARAPHRASE_REVISION = "b521e5ab755a301191c9118c6afc6889895f37bf"
PARALLEL_REVISION = "e4c18f52adbb7dfa4a7aead58c69394b2f446ea6"

NANOBEIR_SUBSETS = {
    "arguana": "NanoArguAna",
    "climatefever": "NanoClimateFEVER",
    "dbpedia": "NanoDBPedia",
    "fever": "NanoFEVER",
    "fiqa2018": "NanoFiQA2018",
    "hotpotqa": "NanoHotpotQA",
    "msmarco": "NanoMSMARCO",
    "nfcorpus": "NanoNFCorpus",
    "nq": "NanoNQ",
    "quoraretrieval": "NanoQuoraRetrieval",
    "scidocs": "NanoSCIDOCS",
    "scifact": "NanoSciFact",
    "touche2020": "NanoTouche2020",
}

SANOIR_DOMAINS = {
    "agriculture": "agriculture",
    "climate-and-environment": "climate and environment",
    "community-discussion": "community discussion",
    "education": "education",
    "energy-and-infrastructure": "energy and infrastructure",
    "health-and-nutrition": "health and nutrition",
    "history-and-culture": "history and culture",
    "law-and-policy": "law and policy",
    "nepal-civic-services": "Nepal civic services",
    "personal-finance": "personal finance",
    "products-and-services": "products and services",
    "science-and-research": "science and research",
    "technology": "technology",
    "travel-and-geography": "travel and geography",
    "work-and-livelihoods": "work and livelihoods",
}

RETRIEVAL_METRIC_PREFIXES = (
    "ndcg_at_",
    "map_at_",
    "mrr_at_",
    "recall_at_",
    "hit_rate_at_",
)


def _metadata(
    *,
    name: str,
    path: str,
    revision: str,
    description: str,
    task_type: str,
    eval_langs: list[str] | dict[str, list[str]],
    main_score: str,
    domains: list[str],
    prompt: str,
) -> TaskMetadata:
    return TaskMetadata(
        name=name,
        dataset={"path": path, "revision": revision},
        description=description,
        reference=f"https://huggingface.co/datasets/{path}/tree/{revision}",
        type=task_type,
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=eval_langs,
        main_score=main_score,
        date=None,
        domains=domains,
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=None,
        prompt=prompt,
        contributed_by="jangedoo",
    )


class STSBNepaliV3(AbsTaskSTS):
    min_score = 0
    max_score = 5
    metadata = _metadata(
        name="STSBNepali.v3",
        path="jangedoo/stsb_nepali",
        revision=STSB_REVISION,
        description="STS-B in Nepali and both English–Nepali directions.",
        task_type="STS",
        eval_langs={
            "ne-ne": ["nep-Deva"],
            "en-ne": ["eng-Latn", "nep-Deva"],
            "ne-en": ["nep-Deva", "eng-Latn"],
        },
        main_score="cosine_spearman",
        domains=["News", "Written"],
        prompt="Retrieve semantically similar text.",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        source = load_dataset(**self.metadata.dataset, split="test", num_proc=num_proc)
        self.dataset = {
            direction: DatasetDict(
                {"test": Dataset.from_list(normalize_sts_rows(source, direction))}
            )
            for direction in self.hf_subsets
        }
        self.data_loaded = True


class _NEBRetrievalTask(AbsTaskRetrieval):
    """Retrieval task reporting only NEB's selected metric families."""

    def _evaluate_subset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        scores = super()._evaluate_subset(*args, **kwargs)
        return {
            name: value
            for name, value in scores.items()
            if name.startswith(RETRIEVAL_METRIC_PREFIXES)
        }


class NanoBEIRNepaliRetrievalV3(AbsTaskRetrieval):
    """Legacy v3 protocol retained only to validate existing evidence."""

    k_values = (1, 3, 5, 10, 20, 100, 1000)
    _top_k = 1000
    metadata = _metadata(
        name="NanoBEIRNepaliRetrieval.v3",
        path="jangedoo/NanoBEIR-ne",
        revision=NANOBEIR_REVISION,
        description="Thirteen separately reported Nepali NanoBEIR retrieval subsets.",
        task_type="Retrieval",
        eval_langs={name: ["nep-Deva"] for name in NANOBEIR_SUBSETS},
        main_score="ndcg_at_10",
        domains=["Web", "Written"],
        prompt="Given a query, retrieve relevant passages.",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        resources = {
            config: load_dataset(
                self.metadata.dataset["path"],
                config,
                revision=self.metadata.dataset["revision"],
                num_proc=num_proc,
            )
            for config in ("corpus", "queries", "qrels")
        }
        self.dataset = {}
        for subset, split in NANOBEIR_SUBSETS.items():
            values = normalize_nanobeir(
                resources["corpus"][split],
                resources["queries"][split],
                resources["qrels"][split],
            )
            self.dataset[subset] = {
                "test": RetrievalSplitData(
                    corpus=Dataset.from_list(values["corpus"]),
                    queries=Dataset.from_list(values["queries"]),
                    relevant_docs=values["relevant_docs"],
                    top_ranked=None,
                )
            }
        self.data_loaded = True


class NanoBEIRNepaliRetrievalV4(_NEBRetrievalTask):
    k_values = (1, 3, 5, 10, 20, 100, 1000)
    _top_k = 1000
    metadata = _metadata(
        name="NanoBEIRNepaliRetrieval.v4",
        path="jangedoo/NanoBEIR-ne",
        revision=NANOBEIR_REVISION,
        description="Thirteen separately reported Nepali NanoBEIR retrieval subsets.",
        task_type="Retrieval",
        eval_langs={name: ["nep-Deva"] for name in NANOBEIR_SUBSETS},
        main_score="ndcg_at_10",
        domains=["Web", "Written"],
        prompt="Given a query, retrieve relevant passages.",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        resources = {
            config: load_dataset(
                self.metadata.dataset["path"],
                config,
                revision=self.metadata.dataset["revision"],
                num_proc=num_proc,
            )
            for config in ("corpus", "queries", "qrels")
        }
        self.dataset = {}
        for subset, split in NANOBEIR_SUBSETS.items():
            values = normalize_nanobeir(
                resources["corpus"][split],
                resources["queries"][split],
                resources["qrels"][split],
            )
            self.dataset[subset] = {
                "test": RetrievalSplitData(
                    corpus=Dataset.from_list(values["corpus"]),
                    queries=Dataset.from_list(values["queries"]),
                    relevant_docs=values["relevant_docs"],
                    top_ranked=None,
                )
            }
        self.data_loaded = True


class NepaliHardNegativesRerankingV3(AbsTaskRetrieval):
    """Legacy reranking protocol retained only to validate existing evidence."""

    k_values = (1, 3, 5)
    _top_k = 5
    metadata = _metadata(
        name="NepaliHardNegativesReranking.v3",
        path="jangedoo/nepali-query-passage-hard-negatives-10k",
        revision=HARD_NEGATIVES_REVISION,
        description="Per-query reranking over one positive and isolated hard negatives.",
        task_type="Reranking",
        eval_langs={"hard-negatives": ["nep-Deva"]},
        main_score="hit_rate_at_1",
        domains=["Web", "Written"],
        prompt="Given a query, retrieve the matching passage.",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        source = load_dataset(**self.metadata.dataset, split="test", num_proc=num_proc)
        values = normalize_hard_negative_rows(source)
        self.dataset = {
            "hard-negatives": {
                "test": RetrievalSplitData(
                    corpus=Dataset.from_list(values["corpus"]),
                    queries=Dataset.from_list(values["queries"]),
                    relevant_docs=values["relevant_docs"],
                    top_ranked=values["top_ranked"],
                )
            }
        }
        self.data_loaded = True


class NepaliHardNegativesRetrievalV4(_NEBRetrievalTask):
    k_values = (1, 3, 5, 10, 20, 100, 1000)
    _top_k = 1000
    metadata = _metadata(
        name="NepaliHardNegativesRetrieval.v4",
        path="jangedoo/nepali-query-passage-hard-negatives-10k",
        revision=HARD_NEGATIVES_REVISION,
        description="Full-corpus retrieval over positives and pooled hard negatives.",
        task_type="Retrieval",
        eval_langs={"hard-negatives": ["nep-Deva"]},
        main_score="ndcg_at_10",
        domains=["Web", "Written"],
        prompt="Given a query, retrieve the matching passage.",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        source = load_dataset(**self.metadata.dataset, split="test", num_proc=num_proc)
        values = normalize_hard_negative_retrieval_rows(source)
        self.dataset = {
            "hard-negatives": {
                "test": RetrievalSplitData(
                    corpus=Dataset.from_list(values["corpus"]),
                    queries=Dataset.from_list(values["queries"]),
                    relevant_docs=values["relevant_docs"],
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True


class NepaliEcommerceRetrievalV1(_NEBRetrievalTask):
    k_values = (1, 3, 5, 10, 20, 100, 1000)
    _top_k = 1000
    metadata = _metadata(
        name="NepaliEcommerceRetrieval.v1",
        path="jangedoo/nepali-ecommerce-retrieval",
        revision=ECOMMERCE_REVISION,
        description="Cross-lingual e-commerce retrieval with pooled supplied negatives.",
        task_type="Retrieval",
        eval_langs={"ecommerce": ["nep-Deva", "nep-Latn", "eng-Latn"]},
        main_score="ndcg_at_10",
        domains=["Web", "Written"],
        prompt="Given a shopping query, retrieve the relevant product passage.",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        source = load_dataset(**self.metadata.dataset, split="test", num_proc=num_proc)
        values = normalize_row_retrieval(
            source,
            positive_column="document",
            negative_columns=("negative1", "negative2", "negative3"),
        )
        self.dataset = {
            "ecommerce": {
                "test": RetrievalSplitData(
                    corpus=Dataset.from_list(values["corpus"]),
                    queries=Dataset.from_list(values["queries"]),
                    relevant_docs=values["relevant_docs"],
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True


class SanoIRGeneralRetrievalV1(_NEBRetrievalTask):
    k_values = (1, 3, 5, 10, 20, 100, 1000)
    _top_k = 1000
    metadata = _metadata(
        name="SanoIRGeneralRetrieval.v1",
        path="jangedoo/sanoir-general",
        revision=SANOIR_REVISION,
        description="Domain-local retrieval across fifteen Nepali and mixed-language domains.",
        task_type="Retrieval",
        eval_langs={subset: ["nep-Deva", "nep-Latn", "eng-Latn"] for subset in SANOIR_DOMAINS},
        main_score="ndcg_at_10",
        domains=["Web", "Written"],
        prompt="Given a query, retrieve the relevant Nepali passage.",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        source = load_dataset(**self.metadata.dataset, split="test", num_proc=num_proc)
        self.dataset = {}
        for subset, domain in SANOIR_DOMAINS.items():
            values = normalize_row_retrieval(
                (row for row in source if row["domain"] == domain),
                positive_column="positive",
                negative_columns=("negative_1", "negative_2", "negative_3", "negative_4"),
            )
            self.dataset[subset] = {
                "test": RetrievalSplitData(
                    corpus=Dataset.from_list(values["corpus"]),
                    queries=Dataset.from_list(values["queries"]),
                    relevant_docs=values["relevant_docs"],
                    top_ranked=None,
                )
            }
        self.data_loaded = True


class NepaliParaphraseClassificationV3(AbsTaskPairClassification):
    label_column_name = "labels"
    metadata = _metadata(
        name="NepaliParaphraseClassification.v3",
        path="jangedoo/paraphrase-nepali",
        revision=PARAPHRASE_REVISION,
        description="Native pair classification for English–Nepali paraphrases.",
        task_type="PairClassification",
        eval_langs={"bilingual": ["eng-Latn", "nep-Deva"]},
        main_score="max_ap",
        domains=["Web", "Written"],
        prompt="Retrieve text that is semantically equivalent to the given text.",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        source = load_dataset(**self.metadata.dataset, num_proc=num_proc)
        self.dataset = {
            "bilingual": DatasetDict(
                {
                    split: dataset.rename_column("label", "labels")
                    if "label" in dataset.column_names
                    else dataset
                    for split, dataset in source.items()
                }
            )
        }
        self.data_loaded = True


class EnglishNepaliBitextMiningV3(AbsTaskBitextMining):
    metadata = _metadata(
        name="EnglishNepaliBitextMining.v3",
        path="jangedoo/en_ne_parallel_corpus",
        revision=PARALLEL_REVISION,
        description="English→Nepali and Nepali→English bitext mining.",
        task_type="BitextMining",
        eval_langs={
            "en-ne": ["eng-Latn", "nep-Deva"],
            "ne-en": ["nep-Deva", "eng-Latn"],
        },
        main_score="f1",
        domains=["Web", "Written"],
        prompt="Retrieve the parallel sentence in the other language.",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        source = load_dataset(**self.metadata.dataset, split="test", num_proc=num_proc)
        self.dataset = {
            direction: DatasetDict(
                {
                    "test": Dataset.from_list(
                        normalize_parallel_direction(source, direction.split("-")[0])
                    )
                }
            )
            for direction in self.hf_subsets
        }
        self.data_loaded = True


def get_tasks() -> list[mteb.AbsTask]:
    """Return the fixed NEB task composition; no membership floats with MTEB."""
    return [
        STSBNepaliV3(),
        NanoBEIRNepaliRetrievalV4(),
        NepaliHardNegativesRetrievalV4(),
        NepaliEcommerceRetrievalV1(),
        SanoIRGeneralRetrievalV1(),
        NepaliParaphraseClassificationV3(),
        EnglishNepaliBitextMiningV3(),
        mteb.get_task("NepaliNewsClassification.v2"),
        mteb.get_task("IndicGenBenchFloresBitextMining", hf_subsets=["nep-eng", "eng-nep"]),
        mteb.get_task(
            "NTREXBitextMining",
            hf_subsets=["nep_Deva-eng_Latn", "eng_Latn-nep_Deva"],
        ),
    ]


def get_result_tasks() -> list[mteb.AbsTask]:
    """Return active tasks plus legacy protocols needed for evidence validation."""
    return [*get_tasks(), NanoBEIRNepaliRetrievalV3(), NepaliHardNegativesRerankingV3()]


def get_benchmark() -> mteb.Benchmark:
    """Return NEB with all benchmark-level aggregations deliberately disabled."""
    return mteb.Benchmark(
        name="NEB(Nepali, v1)",
        display_name="Nepali Embedding Benchmark",
        description="Task-first evaluation of Nepali text embeddings.",
        reference="https://github.com/jangedoo/nepali-embedding-benchmark",
        tasks=get_tasks(),
        language_view=["nep"],
        aggregations=[],
        show_zero_shot=False,
    )
