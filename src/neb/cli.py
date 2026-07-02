"""Command-line interface for NEB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from neb.api import evaluate as evaluate_api
from neb.export import export_static
from neb.registry import load_models, load_tasks, project_root, validate_registries
from neb.results import discover_results, publish_run, validate_run
from neb.scaffold import scaffold_model
from neb.schemas import RuntimeSettings, VerificationStatus

app = typer.Typer(no_args_is_help=True, help="Nepali Embedding Benchmark")
model_app = typer.Typer(no_args_is_help=True, help="Model registry commands")
results_app = typer.Typer(no_args_is_help=True, help="Result lifecycle commands")
app.add_typer(model_app, name="model")
app.add_typer(results_app, name="results")


@app.command("validate")
def validate_command(
    root: Annotated[Path | None, typer.Option(help="Repository root")] = None,
) -> None:
    base = project_root(root)
    tasks, models = validate_registries(base)
    count = 0
    for provenance in sorted((base / "results").glob("*/*/*/*/provenance.json")):
        validate_run(provenance.parent, base)
        count += 1
    typer.echo(f"valid: {len(tasks)} tasks, {len(models)} models, {count} result runs")


@model_app.command("scaffold")
def model_scaffold(
    hf_id: str,
    root: Annotated[Path | None, typer.Option(help="Repository root")] = None,
) -> None:
    typer.echo(scaffold_model(hf_id, project_root(root)))


@app.command("evaluate")
def evaluate_command(
    model: Annotated[str, typer.Option("--model", help="Registered model id or HF id")],
    task: Annotated[list[str] | None, typer.Option("--task", help="Registered task id")] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    device: Annotated[str, typer.Option()] = "cpu",
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    dtype: Annotated[
        str | None,
        typer.Option(help="Model dtype override (bf16, fp16, or fp32)"),
    ] = None,
    allow_remote_code: Annotated[
        bool, typer.Option("--allow-remote-code", help="Allow pinned owner-only model code")
    ] = False,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    root: Annotated[Path | None, typer.Option(help="Repository root")] = None,
) -> None:
    paths = evaluate_api(
        model,
        task,
        runtime=RuntimeSettings(device=device, batch_size=batch_size, dtype=dtype, resume=resume),
        allow_remote_code=allow_remote_code,
        output_dir=output_dir,
        root=project_root(root),
    )
    for path in paths:
        typer.echo(path)


@app.command("queue")
def queue_command(
    as_json: Annotated[bool, typer.Option("--json")] = False,
    root: Annotated[Path | None, typer.Option(help="Repository root")] = None,
) -> None:
    base = project_root(root)
    verified = discover_results(base, include_community=False)
    covered = {(item.model_id, item.task_id, item.view_id) for item in verified}
    queue = [
        {
            "model_id": model.id,
            "task_id": task.id,
            "missing_views": [
                view.id for view in task.views if (model.id, task.id, view.id) not in covered
            ],
        }
        for model in load_models(base)
        for task in load_tasks(base)
        if any((model.id, task.id, view.id) not in covered for view in task.views)
    ]
    if as_json:
        typer.echo(json.dumps(queue, indent=2))
    else:
        for item in queue:
            typer.echo(f"{item['model_id']}\t{item['task_id']}\t{','.join(item['missing_views'])}")


@results_app.command("publish")
def results_publish(
    run: Path,
    status: Annotated[VerificationStatus, typer.Option("--status")],
    skip_existing: Annotated[
        bool, typer.Option("--skip-existing", help="Validate and skip existing publications")
    ] = False,
    root: Annotated[Path | None, typer.Option(help="Repository root")] = None,
) -> None:
    typer.echo(
        publish_run(
            run.resolve(),
            status,
            project_root(root),
            skip_existing=skip_existing,
        )
    )


@app.command("export")
def export_command(
    output: Annotated[Path | None, typer.Option(help="Export directory")] = None,
    root: Annotated[Path | None, typer.Option(help="Repository root")] = None,
) -> None:
    for path in export_static(project_root(root), output):
        typer.echo(path)


if __name__ == "__main__":
    app()
