# NEB — Nepali Embedding Benchmark

NEB is a thin, task-first benchmark built on [MTEB](https://github.com/embeddings-benchmark/mteb/). NEB supplies Nepali task adapters, fixed benchmark composition, evidence publication, static exports, and the dashboard.

You can find the leaderboard at [Nepali Embedding Benchmark](https://jangedoo.github.io/nepali-embedding-benchmark/)

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

Run `neb` in an interactive terminal to open the guided workflow. It walks through active task
and dataset selection, model and revision, device, batch size, precision, cache and prompt
settings, then shows the complete plan before evaluation starts. Afterward, an NEB checkout can
publish just that model revision as community or maintainer-verified evidence and regenerate the
dashboard export.

```bash
neb
```

Use the explicit subcommands below for scripts, CI, and reproducible command history.

Both interfaces default to CUDA, batch size 64, and bfloat16. Override these settings explicitly
when evaluating on CPU, Apple Metal, or hardware without bfloat16 support.

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

Maintainer-verified evidence is CODEOWNERS-protected.

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
