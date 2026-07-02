# NEB — Nepali Embedding Benchmark

NEB helps you compare sentence-embedding models on Nepali and English–Nepali tasks. It provides a
Python package, command-line interface, reproducible evaluation workflow, and static dashboard.

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

## Quick start

You need Python 3.10 or newer and an internet connection for the first model or dataset download.
Install the package in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install nepali-embedding-benchmark

# Confirm that all bundled tasks and models are valid.
neb validate

# See every available command.
neb --help
```

If you want to contribute or run the dashboard, install the repository with
[`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jangedoo/nepali-embedding-benchmark.git
cd nepali-embedding-benchmark
uv sync --locked --extra dev
uv run neb validate
```

The examples below use the installed `neb` command. When working from a source checkout without
activating its virtual environment, prefix it with `uv run`.

To inspect the benchmark from Python:

```python
from neb import get_models, get_tasks

for task in get_tasks():
    print(task.id, [view.id for view in task.views])

for model in get_models():
    print(model.id, model.hf_id)
```

## Run your first evaluation

The following command evaluates the pinned multilingual E5 baseline on the three STS-B Nepali
views:

```bash
neb evaluate \
  --model multilingual-e5-small \
  --task stsb-nepali \
  --device cpu \
  --resume
```

The first run downloads the pinned model and dataset from Hugging Face. CPU evaluation works, but a
CUDA device is recommended for larger or retrieval-heavy runs:

```bash
neb evaluate \
  --model multilingual-e5-small \
  --task nanobeir-ne \
  --device cuda \
  --batch-size 64 \
  --resume
```

Outputs are stored under `runs/`. Each task run contains:

- MTEB-compatible JSON for every evaluation view;
- `model_meta.json` with the exact model revision;
- `run_settings.jsonl` with runtime options;
- `provenance.json` with versions, effective prompts, dataset revision, hardware, and result
  hashes.

`--resume` skips views that are already complete. Some reviewed `jangedoo/*` models contain custom
model code and require an explicit `--allow-remote-code` flag. NEB never enables it implicitly.

The equivalent Python API is:

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
| Nepali hard-negative reranking | One explicit-positive candidate set | MAP@1000 |
| NanoBEIR Nepali | 13 separately reported retrieval subsets | NDCG@10 |
| Nepali paraphrase classification | One bilingual view | Maximum average precision |
| English–Nepali parallel corpus | English→Nepali and Nepali→English | F1 |

Every displayed result has an evidence label:

- **Verified** means a maintainer ran the pinned model and dataset in the pinned environment. It is
  not a claim about training-data contamination.
- **Community** means the submission passed schema, revision, range, and hash checks but was not
  rerun by a maintainer.

When both exist for the same model revision and task view, the dashboard shows the verified result
by default. Missing results remain visible as missing coverage.

## Use the dashboard locally

Node.js 22.12 or newer is required only for dashboard development.

```bash
make sync
make site-install
make site-dev
```

Open the local URL printed by Astro. The dashboard provides task rankings, model search, coverage,
and side-by-side comparison for two to five models.

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
/data/v1/catalog.json
/data/v1/models.json
/data/v1/tasks.json
/data/v1/results.json
/data/v1/results.csv
```

These files are suitable for notebooks, plots, static websites, and Jekyll data visualizations.
Generated files under `site/public/data/v1/` should not be edited by hand.

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

## License

NEB is licensed under Apache-2.0. Models and datasets retain their respective upstream licenses.
