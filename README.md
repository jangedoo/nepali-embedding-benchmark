# NEB — Nepali Embedding Benchmark

NEB helps you compare sentence-embedding models on Nepali and English–Nepali tasks. It provides a
Python package, command-line interface, reproducible evaluation workflow, and static dashboard.

[Explore the published benchmark dashboard](https://jangedoo.github.io/nepali-embedding-benchmark/).

NEB is task-first: models are ranked within a specific task, subset, and metric. It does **not**
combine unrelated scores into a global leaderboard.

## What can I do with NEB?

- Explore model performance on Nepali similarity, retrieval, reranking, paraphrase, and bitext
  tasks.
- Run the same evaluations locally with immutable model and dataset revisions.
- Compare two to five models without hiding missing results.
- Consume versioned JSON or CSV data in another website or analysis.
- Submit a public model or a reproducible community result.

NEB is built on MTEB 2.16.2 and Sentence Transformers.

## Add and evaluate your own model

You need Python 3.10 or newer, [`uv`](https://docs.astral.sh/uv/), and an internet connection for
the first model or dataset download. Clone the repository and install its locked environment:

```bash
git clone https://github.com/jangedoo/nepali-embedding-benchmark.git
cd nepali-embedding-benchmark
make sync
```

Scaffold a registry entry for your public, ungated Sentence Transformers model:

```bash
uv run neb model scaffold owner/model
```

The command prints the generated model ID and registry path. Inspect that YAML manifest, then
validate all registries:

```bash
uv run neb validate
```

Use the printed model ID—not the Hugging Face ID—to evaluate every registered task. CUDA is
recommended for larger or retrieval-heavy runs. CUDA evaluations load models in BF16 when the
GPU supports it and FP16 otherwise:

```bash
uv run neb evaluate --model <generated-id> --device cuda --resume
```

Use `--dtype fp32`, `--dtype bf16`, or `--dtype fp16` to override automatic selection. If an
evaluation still runs out of GPU memory, reduce the encoding batch size with `--batch-size`.

To evaluate only part of the benchmark, repeat `--task` with one or more registered task IDs:

```bash
uv run neb evaluate \
  --model <generated-id> \
  --task stsb-nepali \
  --device cuda \
  --resume
```

Outputs are stored under `runs/`. `--resume` skips views that are already complete. Each task run
contains MTEB-compatible result JSON, the exact model revision, runtime settings, effective
prompts, dataset provenance, and result hashes.

Publish completed v2 runs as explicitly unverified community evidence:

```bash
make publish-community MODEL=<generated-id>
```

This copies the model's v2 runs into `results/community/` and regenerates the dashboard exports.
Install dashboard dependencies once, then start the local dashboard to inspect the
community-labeled results:

```bash
make site-install
make site-dev
```

Astro prints the local URL. To regenerate only the versioned files under
`site/public/data/v2/`, run:

```bash
make export
```

Scaffolding is revision-aware. Re-running it for the same Hugging Face ID and commit is an
idempotent no-op. A different commit creates a revision-qualified model ID while preserving the
old registration and results. If several revisions are registered, evaluation by Hugging Face ID
is rejected as ambiguous; use one of the printed registry model IDs.

Some reviewed `jangedoo/*` models contain custom model code and require an explicit
`--allow-remote-code` evaluation flag. NEB never enables it implicitly.

## Python API

To inspect the benchmark from Python:

```python
from neb import get_models, get_tasks

for task in get_tasks():
    print(task.id, [view.id for view in task.views])

for model in get_models():
    print(model.id, model.hf_id)
```

The equivalent evaluation API is:

```python
from neb import evaluate
from neb.schemas import RuntimeSettings

run_directories = evaluate(
    "multilingual-e5-small",
    tasks=["stsb-nepali"],
    runtime=RuntimeSettings(device="cpu", batch_size=32, resume=True),
)
```

## Understand the results

Always compare scores from the same task view and metric. For example, an NDCG@10 retrieval score
and a Spearman correlation score answer different questions and must not be averaged.

NEB currently includes:

| Task | Views | Primary metric |
| --- | --- | --- |
| STS-B Nepali | Nepali–Nepali, English–Nepali, Nepali–English | Cosine Spearman |
| Nepali hard-negative reranking | One explicit-positive candidate set | Hit rate@1 |
| NanoBEIR Nepali | 13 separately reported retrieval subsets | NDCG@10 |
| Nepali paraphrase classification | One bilingual view | Cosine average precision |
| English–Nepali parallel corpus | English→Nepali and Nepali→English | F1 |

Community results carry a yellow evidence icon; verified results are intentionally unmarked:

- **Verified** means a maintainer ran the pinned model and dataset in the pinned environment. It is
  not a claim about training-data contamination.
- **Community** means the submission passed schema, revision, range, and hash checks but was not
  rerun by a maintainer.

When both exist for the same model revision and task view, the dashboard shows the verified result
by default. Missing results remain visible as missing coverage.

## Use the dashboard locally

Node.js 22.12 or newer is required only for dashboard development.

```bash
make site-install
make site-dev
```

Open the local URL printed by Astro. The dashboard provides task rankings with all metrics visible
by default, per-view metric selection, model search, coverage, and side-by-side comparison for two
to five models. The comparison view shows selected metrics as compact per-model cells and provides
global dataset and per-dataset metric filters, all selected by default. Model sizes are measured
during evaluation and remain `unknown` until results are published.

To build the production site:

```bash
make site-build
```

For a GitHub Pages project subpath:

```bash
make site-build BASE_PATH=/nepali-embedding-benchmark/
```

## Use the exported data

Run `make export` to regenerate the public, versioned artifacts from canonical registries and
results:

```text
/data/v2/catalog.json
/data/v2/models.json
/data/v2/tasks.json
/data/v2/results.json
/data/v2/results.csv
```

These files are suitable for notebooks, plots, static websites, and Jekyll data visualizations.
JSON contains one result per model/task/view with a metric map; CSV is long-form with one row per
metric and an `is_primary` column.
Generated files under `site/public/data/v2/` should not be edited by hand.

## Common commands

```bash
make help            # List all development commands.
make check           # Lint, test, and validate the Python project.
make queue           # Show model/task pairs missing verified results.
make publish-verified MODEL=all-minilm-l6-v2-nepali
                      # Publish this model's runs and refresh dashboard data.
make site-check      # Test, build, and audit the dashboard.
make test-contracts  # Check live pinned Hugging Face dataset contracts.
make package         # Build the Python source distribution and wheel.
```

The main CLI commands are:

```text
neb validate
neb model scaffold <hugging-face-id>
neb evaluate --model <id> [--task <id>] --resume
neb queue
neb results publish <run-directory> --status community|verified
neb export
```

## Contribute

Contributions are welcome. You can:

- propose a public Sentence Transformers model;
- submit community evaluation results;
- add a pinned dataset and task definition;
- improve adapters, tests, documentation, or the dashboard.

Start with [CONTRIBUTING.md](CONTRIBUTING.md). It explains model requirements, result publication,
verified-result policy, dataset versioning, and the remote-code boundary. Automated pull-request
checks validate submissions without downloading submitted model weights.

If you are using an AI coding agent in this repository, also read [AGENTS.md](AGENTS.md).

## Reproducibility and scope

- Every model and dataset is pinned to a full Hugging Face commit SHA.
- Score-affecting task changes create a new task version.
- Model-native prompts are used unless a manifest defines an override; effective prompts are
  recorded in provenance.
- Partial model coverage is allowed.
- Unknown upstream metadata is shown as `unknown` rather than guessed.
- NEB does not currently detect or filter training-data contamination.

## Citation

If you use NEB in research or evaluation, please cite:

> Sanjaya Subedi. *NEB — Nepali Embedding Benchmark*. 2026.
> https://github.com/jangedoo/nepali-embedding-benchmark

```bibtex
@software{subedi2026neb,
  author = {Subedi, Sanjaya},
  title = {NEB: Nepali Embedding Benchmark},
  year = {2026},
  url = {https://github.com/jangedoo/nepali-embedding-benchmark}
}
```

## License

NEB is licensed under Apache-2.0. Models and datasets retain their respective upstream licenses.
