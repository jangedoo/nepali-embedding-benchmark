# NEB — Nepali Embedding Benchmark

NEB is a thin, task-first benchmark built on [MTEB 2.18.3](https://github.com/embeddings-benchmark/mteb/releases/tag/2.18.3). MTEB owns model loading, prompts, encoding, evaluation, metrics, caching, and `TaskResult` JSON. NEB supplies Nepali task adapters, fixed benchmark composition, evidence publication, static exports, and the dashboard.

NEB intentionally has no global score or overall model ranking. Compare models only inside a task, subset, split, and metric view. Partial task coverage is valid and remains visible.

## Benchmark composition

`neb.get_benchmark()` returns the MTEB benchmark `NEB(Nepali, v1)` with all benchmark aggregation disabled. It contains:

- `STSBNepali.v3`
- `NanoBEIRNepaliRetrieval.v3`
- `NepaliHardNegativesReranking.v3`
- `NepaliParaphraseClassification.v3`
- `EnglishNepaliBitextMining.v3`
- `NepaliNewsClassification.v2`
- `IndicGenBenchFloresBitextMining` (`nep-eng`, `eng-nep` only)
- `NTREXBitextMining` (Nepali↔English only)

All dataset revisions are full Hugging Face commit SHAs. The five NEB-owned tasks use small pure transforms in `src/neb/adapters.py`; there is no task-manifest DSL.

## Install and verify

```bash
make sync
make check
make site-check
make package
```

Pinned network contract tests are opt-in:

```bash
make test-contracts
```

## Evaluate a model

By default NEB resolves the current Hugging Face Hub HEAD and records its exact 40-character
commit SHA:

```bash
neb run \
  --model owner/model \
  --device cuda \
  --batch-size 64 \
  --task STSBNepali.v3
```

Pass `--revision 0123456789abcdef0123456789abcdef01234567` when a run must be pinned
before resolution. `--task` is repeatable. Results resume with MTEB's `only-missing` behavior
in `ResultCache(runs/)`. Optional `--dtype`, `--query-prompt`, and `--document-prompt` values
are passed through the native model wrapper.

An existing local SentenceTransformer directory is also accepted:

```bash
neb run --model ./checkpoints/my-model --device cuda --task STSBNepali.v3
```

NEB hashes the local directory contents into a deterministic `local-<sha256>` cache revision.
Local results are useful for development and resume normally, but cannot be published as
canonical evidence or exported to the dashboard. `--revision` and `--allow-remote-code` are not
accepted for local paths.

Exact model behavior missing upstream is configured in `registries/models/*.yaml`, never in a
Python preset map. Prompt keys follow MTEB directly: `query` and `document` select asymmetric
roles, task types such as `STS` apply to symmetric task families, exact task names target one
task, and compound keys such as `Retrieval-query` are most specific. Values are passed unchanged
to the model loader. Native `passage` keys are not reinterpreted as `document`.

`neb run` logs revision resolution, metadata and prompt sources, effective prompts, loader and
runtime settings, and per-task MTEB prompt selection to stderr. Use `--log-level` to adjust the
verbosity; the final machine-readable summary remains on stdout.

Remote code is disabled by default. It can only be enabled with `--allow-remote-code` for an
approved, exact `jangedoo/*` YAML override. CI never evaluates submitted weights or remote code.

The Python API exposes native MTEB objects:

```python
from neb import evaluate, get_benchmark, get_tasks, resolve_model

benchmark = get_benchmark()       # mteb.Benchmark, aggregations=[]
tasks = get_tasks()               # list[mteb.AbsTask]
meta = resolve_model("owner/model")            # mteb.ModelMeta with resolved exact SHA
result = evaluate(meta, tasks=[tasks[0]])        # mteb.ModelResult
```

## Evidence lifecycle

Evaluation caches contain untouched MTEB task JSON, `model_meta.json`, `run_settings.jsonl`, and adjacent `.json.sha256` integrity files. Publish a cache or individual task file with:

```bash
neb results publish runs --status community
neb results publish runs --status verified
```

Canonical caches live below `results/community/` and `results/verified/`. Community evidence is explicitly unverified. Verified evidence takes precedence per model revision, task, split, and subset; it never relabels unrelated community coverage. Existing canonical task files may gain missing splits or subsets, but conflicting scores are rejected.

To intentionally replace a rerun's colliding scores and run settings, pass `--overwrite`. If
the model loader metadata changed (for example prompts, dtype, or device), the source must be a
complete cache covering every score already published for that model revision so retained scores
cannot be relabeled with different evaluation settings:

```bash
neb results publish runs --status verified --overwrite
```

Maintainer-verified evidence is CODEOWNERS-protected. Real scores require the maintainer's local GPU environment and must never be fabricated.

## Static dashboard contract

Run `make export` after canonical evidence changes. This generates, but should never be hand-edited:

```text
site/public/data/v3/catalog.json
site/public/data/v3/models.json
site/public/data/v3/tasks.json
site/public/data/v3/results.json
site/public/data/v3/results.csv
```

The dashboard groups model revisions by Hugging Face repository, defaults to the most recently evaluated canonical revision, provides a global history toggle, links exact model and dataset revisions, and exposes full evidence details. Rankings are task-local and show the native main score before optional secondary metrics.
