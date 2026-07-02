# AGENTS.md

This repository contains NEB, the Nepali Embedding Benchmark. Changes must preserve its
task-first design: never introduce a global score or an overall model ranking.

## Repository map

- `src/neb/`: Python package, CLI, schemas, adapters, evaluation, and result lifecycle.
- `registries/tasks/`: versioned task manifests with immutable dataset revisions.
- `registries/models/`: model manifests with immutable model revisions.
- `results/verified/`: maintainer-verified canonical results; CODEOWNERS-protected.
- `results/community/`: schema-checked, explicitly unverified community results.
- `site/`: Astro and React dashboard.
- `site/public/data/v2/`: generated exports; regenerate with `make export`.
- `tests/`: Python unit and opt-in network contract tests.

## Development commands

Run `make help` for the complete command list. The usual verification loop is:

```bash
make sync
make check
make site-check
make package
```

Dataset contract tests access pinned Hugging Face revisions and are intentionally opt-in:

```bash
make test-contracts
```

## Invariants

- Pin Hugging Face models and datasets to full 40-character commit SHAs.
- Increment a task version for score-affecting dataset, split, transformation, or metric changes.
- Do not increment a task version for metadata-only changes.
- Keep unusual dataset transformations small, pure, reviewed, and unit-tested in
  `src/neb/adapters.py`.
- Preserve MTEB-compatible per-task result JSON and SHA-256 provenance.
- Verified results take precedence over community results, but community evidence must never be
  relabeled as verified.
- `trust_remote_code` is restricted to exact pinned `jangedoo/*` revisions and requires the
  explicit local CLI flag. Pull-request CI must never load submitted model weights or remote code.
- Use model-native prompts unless a manifest supplies a query/document override; always record
  effective prompts.
- Partial coverage is valid and must remain visible in exports and the dashboard.
- Missing metadata is represented as `unknown`; do not invent upstream metadata.

## Editing and generated files

- Edit registry YAML, canonical results, Python source, and dashboard source directly.
- Do not hand-edit files under `site/public/data/v2/`; run `make export` after source-data changes.
- Do not commit `runs/`, `site/dist/`, `site/node_modules/`, caches, or virtual environments.
- Keep Python dependencies locked in `uv.lock` and dashboard dependencies locked in
  `site/package-lock.json`.
- When changing Python dependencies, run `uv lock`. When changing dashboard dependencies, run
  `make site-install` and commit the updated lockfile.

## Testing expectations

- Registry/schema changes require validation tests.
- Adapter changes require fixture-based transformation tests.
- Result lifecycle changes require malformed input, hash, revision, duplicate, and precedence
  coverage.
- Dashboard changes require Vitest coverage where practical and must retain accessible labels,
  keyboard-operable controls, evidence badges, missing-result states, and subpath-safe URLs.
- Before handoff, run `make check site-check package` and report any intentionally skipped network
  or GPU evaluation.

Real benchmark scores require the maintainer's local evaluation environment. Do not fabricate
results or mark a run verified merely to satisfy coverage targets.
