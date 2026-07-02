# Contributing

## Add a model

Run `neb model scaffold owner/model`, inspect the pinned manifest, then open a pull request. A
community model must be public, ungated, loadable by `SentenceTransformer`, and must not require
remote code. Merging a model adds it to the pending queue; CI never downloads submitted weights.

## Submit a community result

Evaluate an existing pinned manifest, keep the generated MTEB JSON and provenance intact, then run
`neb results publish <run-directory> --status community`. Submit only that new canonical result
directory. CI verifies schemas, metric ranges, model/dataset revisions, duplicate submissions, and
SHA-256 hashes. Community evidence is always labeled `community`.

## Maintainer evaluation

Use the lockfile environment and a local GPU, run `neb queue`, then evaluate missing pairs with
`--resume`. Inspect the effective prompts and hardware provenance before publishing with
`--status verified` in a separate result pull request. CODEOWNERS review is required for the
verified tree. Gated bases are excluded. Owner-only remote code may be run locally with the explicit
flag, but it is never run in pull-request CI.

## Add or change a dataset

Standard STS, retrieval, reranking, pair-classification, and bitext datasets need a YAML task
manifest and contract fixtures. Unusual transformations require a small, pure adapter in
`src/neb/adapters.py`, focused unit tests, and review. Pin the full dataset commit. A change to the
dataset, transformation, evaluation split, or metric increments `TaskSpec.version`; metadata-only
changes do not. Missing dataset-card metadata is represented as `unknown` and does not block a task.

NEB has no contamination declaration or detection system. Do not present verified execution as a
contamination audit.

