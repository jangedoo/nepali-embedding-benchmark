"""Local, pinned evaluation workflow.

Heavy evaluation dependencies are imported only after registry and policy checks, so
validation, queueing, and static exports stay usable on CPU-only machines.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neb.adapters import (
    add_nanobeir_relevance,
    explicit_positive_candidates,
    normalize_parallel_direction,
)
from neb.registry import load_models, load_tasks
from neb.schemas import (
    AdapterKind,
    ModelSpec,
    PromptOverrides,
    RunProvenance,
    RuntimeSettings,
    TaskSpec,
    TaskView,
    VerificationStatus,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(items: list[Any], identifier: str, kind: str) -> Any:
    by_id = {item.id: item for item in items}
    by_hf = {getattr(item, "hf_id", ""): item for item in items}
    if identifier in by_id:
        return by_id[identifier]
    if identifier in by_hf:
        return by_hf[identifier]
    raise ValueError(f"unknown {kind} {identifier!r}")


def resolve_model_dtype(runtime: RuntimeSettings, torch: Any) -> tuple[RuntimeSettings, Any | None]:
    """Resolve the requested model dtype and return settings suitable for provenance."""
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    requested = runtime.dtype.lower() if runtime.dtype is not None else None
    if requested is None and runtime.device.lower().split(":", 1)[0] == "cuda":
        requested = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if requested is None:
        return runtime, None
    try:
        resolved = aliases[requested]
    except KeyError as exc:
        choices = ", ".join(sorted(aliases))
        raise ValueError(f"unsupported dtype {runtime.dtype!r}; choose one of: {choices}") from exc
    return runtime.model_copy(update={"dtype": resolved}), getattr(torch, resolved)


def sts_metrics(similarities: Sequence[float], gold: Sequence[float]) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr

    if len(similarities) != len(gold) or len(gold) < 2:
        raise ValueError("STS scores must contain at least two equally shaped samples")
    return {
        "cosine_spearman": float(spearmanr(similarities, gold).statistic),
        "cosine_pearson": float(pearsonr(similarities, gold).statistic),
    }


def retrieval_metrics(labels: Any, scores: Any, *, k: int = 10) -> dict[str, float]:
    """Calculate retrieval metrics with the pinned MTEB implementation."""
    import os
    import tempfile

    previous_cache = os.environ.get("MTEB_CACHE")
    os.environ.setdefault("MTEB_CACHE", str(Path(tempfile.gettempdir()) / "neb-mteb-cache"))
    try:
        from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores
    finally:
        if previous_cache is None:
            del os.environ["MTEB_CACHE"]

    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("retrieval labels and scores must be equally shaped matrices")
    qrels = {
        f"q{query}": {
            f"d{document}": int(relevance)
            for document, relevance in enumerate(relevant)
            if relevance > 0
        }
        for query, relevant in enumerate(labels)
    }
    qrels = {query: relevant for query, relevant in qrels.items() if relevant}
    if not qrels:
        raise ValueError("retrieval evaluation has no queries with relevance judgments")
    results = {
        query: {
            f"d{document}": float(score)
            for document, score in enumerate(scores[int(query.removeprefix("q"))])
        }
        for query in qrels
    }
    evaluated = calculate_retrieval_scores(results, qrels, [k])
    return {
        "ndcg_at_10": evaluated.ndcg[f"NDCG@{k}"],
        "map_at_10": evaluated.map[f"MAP@{k}"],
        "mrr_at_10": evaluated.mrr[f"MRR@{k}"],
        "recall_at_10": evaluated.recall[f"Recall@{k}"],
        "precision_at_10": evaluated.precision[f"P@{k}"],
        "hit_rate_at_10": evaluated.hit_rate[f"HitRate@{k}"],
    }


def reranking_metrics(ranks: Sequence[int]) -> dict[str, float]:
    """Calculate one-positive reranking metrics from one-indexed positive ranks."""
    import numpy as np

    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("reranking ranks must be non-empty positive integers")
    return {
        "hit_rate_at_1": float(np.mean([rank <= 1 for rank in ranks])),
        "mrr_at_5": float(np.mean([1.0 / rank if rank <= 5 else 0.0 for rank in ranks])),
        "ndcg_at_5": float(
            np.mean([1.0 / math.log2(rank + 1) if rank <= 5 else 0.0 for rank in ranks])
        ),
        "hit_rate_at_3": float(np.mean([rank <= 3 for rank in ranks])),
    }


def select_classification_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Select the cosine threshold with maximum accuracy on validation data."""
    import numpy as np
    from sklearn.metrics import accuracy_score

    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if labels_array.shape != scores_array.shape or not len(labels_array):
        raise ValueError("classification labels and scores must be non-empty and equally shaped")
    unique = np.unique(scores_array)
    candidates = [
        float(np.nextafter(unique[0], -np.inf)),
        *[float((left + right) / 2) for left, right in zip(unique[:-1], unique[1:], strict=True)],
        float(np.nextafter(unique[-1], np.inf)),
    ]
    return max(
        candidates,
        key=lambda threshold: (accuracy_score(labels_array, scores_array >= threshold), threshold),
    )


def pair_classification_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = [score >= threshold for score in scores]
    return {
        "cosine_ap": float(average_precision_score(labels, scores)),
        "cosine_roc_auc": float(roc_auc_score(labels, scores)),
        "cosine_accuracy": float(accuracy_score(labels, predictions)),
        "cosine_f1": float(f1_score(labels, predictions, zero_division=0)),
        "cosine_precision": float(precision_score(labels, predictions, zero_division=0)),
        "cosine_recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def bitext_metrics(predictions: Sequence[int], expected: Sequence[int]) -> dict[str, float]:
    """Score directed bitext matches as predicted source-target links."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predicted = np.asarray(predictions)
    truth = np.asarray(expected)
    if predicted.shape != truth.shape or not len(truth):
        raise ValueError(
            "bitext predictions and expected links must be non-empty and equally shaped"
        )
    return {
        "f1": float(f1_score(truth, predicted, zero_division=0, average="weighted")),
        "precision": float(precision_score(truth, predicted, zero_division=0, average="weighted")),
        "recall": float(recall_score(truth, predicted, zero_division=0, average="weighted")),
        "accuracy": float(accuracy_score(truth, predicted)),
    }


class EvaluationRunner:
    def __init__(self, root: Path):
        self.root = root

    def run(
        self,
        model_id: str,
        task_ids: list[str] | None,
        *,
        runtime: RuntimeSettings,
        allow_remote_code: bool,
        output_dir: Path | None,
    ) -> list[Path]:
        model_spec: ModelSpec = _lookup(load_models(self.root), model_id, "model")
        if model_spec.trust_remote_code and not allow_remote_code:
            raise PermissionError(
                f"{model_spec.id} requires remote code; pass --allow-remote-code explicitly"
            )
        all_tasks = load_tasks(self.root)
        tasks = (
            [_lookup(all_tasks, task_id, "task") for task_id in task_ids] if task_ids else all_tasks
        )
        try:
            import sentence_transformers
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("evaluation dependencies are missing; install the package") from exc

        if importlib.metadata.version("mteb") != "2.16.2":
            raise RuntimeError("NEB evaluations require exactly mteb 2.16.2")
        runtime, model_dtype = resolve_model_dtype(runtime, torch)
        model_kwargs = {"torch_dtype": model_dtype} if model_dtype is not None else None
        encoder = SentenceTransformer(
            model_spec.hf_id,
            revision=model_spec.revision,
            trust_remote_code=model_spec.trust_remote_code,
            device=runtime.device,
            model_kwargs=model_kwargs,
        )
        parameter_count, vocab_size = self._model_stats(encoder)
        effective_prompts = self._effective_prompts(model_spec, encoder)
        base = output_dir or self.root / "runs"
        run_id = f"{model_spec.id}-{model_spec.revision[:8]}"
        produced: list[Path] = []
        for task in tasks:
            run_dir = base / run_id / f"{task.id}-v{task.version}"
            expected = [run_dir / "results" / f"{view.id}.json" for view in task.views]
            if runtime.resume and expected and all(path.is_file() for path in expected):
                produced.append(run_dir)
                continue
            (run_dir / "results").mkdir(parents=True, exist_ok=True)
            result_paths: list[Path] = []
            for view in task.views:
                metrics = self._evaluate_view(encoder, model_spec, task, view, runtime)
                result_path = run_dir / "results" / f"{view.id}.json"
                result_path.write_text(
                    json.dumps(
                        self._mteb_result(model_spec, task, view, metrics),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                result_paths.append(result_path)
            provenance = RunProvenance(
                run_id=run_id,
                status=VerificationStatus.community,
                model_id=model_spec.id,
                model_hf_id=model_spec.hf_id,
                model_revision=model_spec.revision,
                task_id=task.id,
                task_version=task.version,
                dataset_revision=task.dataset.revision,
                neb_version="0.2.0",
                sentence_transformers_version=sentence_transformers.__version__,
                parameter_count=parameter_count,
                vocab_size=vocab_size,
                effective_prompts=effective_prompts,
                runtime=runtime,
                hardware={"platform": platform.platform(), "python": platform.python_version()},
                command=sys.argv,
                result_hashes={
                    str(path.relative_to(run_dir)): sha256_file(path) for path in result_paths
                },
            )
            (run_dir / "provenance.json").write_text(
                provenance.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            (run_dir / "model_meta.json").write_text(
                model_spec.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            with (run_dir / "run_settings.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(runtime.model_dump_json() + "\n")
            produced.append(run_dir)
        return produced

    @staticmethod
    def _effective_prompts(spec: ModelSpec, encoder: Any) -> PromptOverrides:
        native = getattr(encoder, "prompts", {}) or {}
        return PromptOverrides(
            query=spec.prompts.query or native.get("query"),
            document=spec.prompts.document or native.get("document") or native.get("passage"),
        )

    @staticmethod
    def _model_stats(encoder: Any) -> tuple[int, int]:
        parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
        tokenizer = getattr(encoder, "tokenizer", None)
        if tokenizer is None and hasattr(encoder, "_first_module"):
            tokenizer = getattr(encoder._first_module(), "tokenizer", None)
        if parameter_count <= 0 or tokenizer is None:
            raise ValueError("loaded model must expose parameters and a tokenizer")
        vocab_size = len(tokenizer)
        if vocab_size <= 0:
            raise ValueError("loaded model tokenizer must have a non-empty vocabulary")
        return parameter_count, vocab_size

    @staticmethod
    def _encode(
        encoder: Any,
        texts: list[str],
        kind: str,
        spec: ModelSpec,
        runtime: RuntimeSettings,
    ) -> Any:
        override = spec.prompts.query if kind == "query" else spec.prompts.document
        kwargs = {
            "batch_size": runtime.batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": False,
            **runtime.encode_kwargs,
        }
        if override is not None:
            kwargs["prompt"] = override
            return encoder.encode(texts, **kwargs)
        method = getattr(encoder, f"encode_{kind}", encoder.encode)
        return method(texts, **kwargs)

    def _evaluate_view(
        self,
        encoder: Any,
        spec: ModelSpec,
        task: TaskSpec,
        view: TaskView,
        runtime: RuntimeSettings,
    ) -> dict[str, float]:
        try:
            import numpy as np
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MTEB evaluation dependencies are incomplete") from exc

        def encode(texts: list[str], kind: str = "document") -> Any:
            values = self._encode(encoder, texts, kind, spec, runtime)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            return values / np.maximum(norms, 1e-12)

        if task.adapter == AdapterKind.retrieval:
            resources = view.resources
            corpus = load_dataset(
                task.dataset.id,
                resources["corpus"],
                split=view.split,
                revision=task.dataset.revision,
            )
            queries = load_dataset(
                task.dataset.id,
                resources["queries"],
                split=view.split,
                revision=task.dataset.revision,
            )
            qrels = add_nanobeir_relevance(
                load_dataset(
                    task.dataset.id,
                    resources["qrels"],
                    split=view.split,
                    revision=task.dataset.revision,
                )
            )
            corpus_ids = [row["_id"] for row in corpus]
            query_ids = [row["_id"] for row in queries]
            scores = (
                encode([row["text"] for row in queries], "query")
                @ encode([row["text"] for row in corpus], "document").T
            )
            cindex = {value: index for index, value in enumerate(corpus_ids)}
            qindex = {value: index for index, value in enumerate(query_ids)}
            labels = np.zeros_like(scores)
            for row in qrels:
                if row["query-id"] in qindex and row["corpus-id"] in cindex:
                    labels[qindex[row["query-id"]], cindex[row["corpus-id"]]] = row["score"]
            return retrieval_metrics(labels, scores, k=10)

        rows = load_dataset(
            task.dataset.id,
            view.config,
            split=view.split,
            revision=task.dataset.revision,
        )
        columns = view.columns
        if task.adapter == AdapterKind.sts:
            left = encode([row[columns["sentence1"]] for row in rows])
            right = encode([row[columns["sentence2"]] for row in rows])
            similarities = np.sum(left * right, axis=1)
            gold = [row[columns["score"]] for row in rows]
            return sts_metrics(similarities, gold)
        if task.adapter == AdapterKind.pair_classification:
            validation = load_dataset(
                task.dataset.id,
                view.config,
                split="valid",
                revision=task.dataset.revision,
            )
            validation_left = encode([row[columns["sentence1"]] for row in validation])
            validation_right = encode([row[columns["sentence2"]] for row in validation])
            validation_scores = np.sum(validation_left * validation_right, axis=1)
            threshold = select_classification_threshold(
                [row[columns["label"]] for row in validation], validation_scores
            )
            left = encode([row[columns["sentence1"]] for row in rows])
            right = encode([row[columns["sentence2"]] for row in rows])
            similarities = np.sum(left * right, axis=1)
            labels = [row[columns["label"]] for row in rows]
            return pair_classification_metrics(labels, similarities, threshold)
        if task.adapter == AdapterKind.reranking:
            samples = explicit_positive_candidates(
                rows,
                query=columns["query"],
                positive=columns["positive"],
                negatives=columns["negatives"],
            )
            ranks: list[int] = []
            for sample in samples:
                candidates = [*sample["positive"], *sample["negative"]]
                values = encode([sample["query"]], "query") @ encode(candidates).T
                order = np.argsort(-values[0])
                rank = int(np.where(order == 0)[0][0]) + 1
                ranks.append(rank)
            return reranking_metrics(ranks)
        if task.adapter == AdapterKind.bitext_mining:
            pairs = normalize_parallel_direction(rows, view.languages[0])
            source = encode([row["sentence1"] for row in pairs], "query")
            target = encode([row["sentence2"] for row in pairs], "document")
            predictions = np.argmax(source @ target.T, axis=1)
            return bitext_metrics(predictions, np.arange(len(pairs)))
        raise ValueError(f"unsupported adapter {task.adapter}")

    @staticmethod
    def _mteb_result(
        model: ModelSpec, task: TaskSpec, view: TaskView, metrics: dict[str, float]
    ) -> dict[str, Any]:
        if set(metrics) != set(view.metrics):
            raise ValueError(f"metrics do not match manifest for {task.id}/{view.id}")
        if any(not math.isfinite(value) or not -1 <= value <= 1 for value in metrics.values()):
            raise ValueError(f"invalid metric value for {task.id}/{view.id}")
        ordered_metrics = {name: metrics[name] for name in view.metrics}
        return {
            "dataset_revision": task.dataset.revision,
            "mteb_dataset_name": task.dataset.id,
            "mteb_version": "2.16.2",
            "model_name": model.hf_id,
            "model_revision": model.revision,
            "task_name": task.id,
            "task_revision": str(task.version),
            "scores": {
                view.split: [
                    {
                        "hf_subset": view.id,
                        "languages": view.languages,
                        "main_score": ordered_metrics[view.primary_metric],
                        **ordered_metrics,
                    }
                ]
            },
        }
