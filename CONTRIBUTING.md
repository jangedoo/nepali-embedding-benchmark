# Contributing

NEB is task-first: do not add a global score, benchmark aggregate, or overall model ranking.

Run `make check`, `make site-check`, and `make package` before handoff. Network dataset contracts are opt-in with `make test-contracts`; model evaluation remains a maintainer GPU workflow.

Task changes belong in native MTEB task classes under `src/neb/tasks.py`. Keep dataset transformations pure and fixture-tested in `src/neb/adapters.py`, pin every dataset revision to a full 40-character SHA, and bump the task name version for score-affecting changes.

Do not edit native MTEB task JSON or `site/public/data/v3/` by hand. Publish evaluated caches with `neb results publish`, then regenerate exports using `make export`. Community submissions remain explicitly unverified; only maintainers may publish verified evidence.

Models are addressed directly by public Hugging Face repository. An omitted revision resolves to
the current Hub commit; reproducible submissions still record an exact 40-character SHA. Add an
exact entry under `registries/models/` only when a revision needs prompt overrides or approved
`jangedoo/*` remote code. Keep these files behavioral rather than using them as a model catalog.
Never load submitted model weights or remote code in pull-request CI.

Model prompt keys use MTEB's native role, task-type, task-name, and compound-key conventions.
Do not rename `passage` to `document`; a model or override must provide the correct `document`
key. Local model directories may be evaluated for development but their fingerprinted results
cannot be published.
