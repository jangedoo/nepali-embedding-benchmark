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
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("evaluation dependencies are missing; install the package") from exc

        if importlib.metadata.version("mteb") != "2.16.2":
            raise RuntimeError("NEB evaluations require exactly mteb 2.16.2")
        encoder = SentenceTransformer(
            model_spec.hf_id,
            revision=model_spec.revision,
            trust_remote_code=model_spec.trust_remote_code,
            device=runtime.device,
        )
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
                score = self._evaluate_view(encoder, model_spec, task, view, runtime)
                result_path = run_dir / "results" / f"{view.id}.json"
                result_path.write_text(
                    json.dumps(
                        self._mteb_result(model_spec, task, view, score),
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
                neb_version="0.1.0",
                sentence_transformers_version=sentence_transformers.__version__,
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
    ) -> float:
        try:
            import numpy as np
            from datasets import load_dataset
            from scipy.stats import spearmanr
            from sklearn.metrics import average_precision_score, ndcg_score
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
            return float(ndcg_score(labels, scores, k=10))

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
            return float(spearmanr(similarities, [row[columns["score"]] for row in rows]).statistic)
        if task.adapter == AdapterKind.pair_classification:
            left = encode([row[columns["sentence1"]] for row in rows])
            right = encode([row[columns["sentence2"]] for row in rows])
            similarities = np.sum(left * right, axis=1)
            labels = [row[columns["label"]] for row in rows]
            return float(
                max(
                    average_precision_score(labels, similarities),
                    average_precision_score(labels, -similarities),
                )
            )
        if task.adapter == AdapterKind.reranking:
            samples = explicit_positive_candidates(
                rows,
                query=columns["query"],
                positive=columns["positive"],
                negatives=columns["negatives"],
            )
            reciprocal_ranks: list[float] = []
            for sample in samples:
                candidates = [*sample["positive"], *sample["negative"]][:1000]
                values = encode([sample["query"]], "query") @ encode(candidates).T
                order = np.argsort(-values[0])
                rank = int(np.where(order == 0)[0][0]) + 1
                reciprocal_ranks.append(1.0 / rank)
            return float(np.mean(reciprocal_ranks))
        if task.adapter == AdapterKind.bitext_mining:
            pairs = normalize_parallel_direction(rows, view.languages[0])
            source = encode([row["sentence1"] for row in pairs], "query")
            target = encode([row["sentence2"] for row in pairs], "document")
            predictions = np.argmax(source @ target.T, axis=1)
            return float(np.mean(predictions == np.arange(len(pairs))))
        raise ValueError(f"unsupported adapter {task.adapter}")

    @staticmethod
    def _mteb_result(
        model: ModelSpec, task: TaskSpec, view: TaskView, score: float
    ) -> dict[str, Any]:
        if not math.isfinite(score):
            raise ValueError(f"non-finite score for {task.id}/{view.id}")
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
                        "main_score": score,
                        view.primary_metric: score,
                    }
                ]
            },
        }
