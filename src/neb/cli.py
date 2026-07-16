"""Command-line interface for the native-MTEB NEB layer."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from neb.schemas import VerificationStatus

app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    help="Nepali Embedding Benchmark",
)
results_app = typer.Typer(no_args_is_help=True, help="Native MTEB evidence commands")
app.add_typer(results_app, name="results")


def project_root(start: Path | None = None) -> Path:
    cursor = (start or Path.cwd()).resolve()
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/neb").is_dir():
            return candidate
    return cursor


@app.callback()
def main(ctx: typer.Context) -> None:
    """Launch the guided workflow when no subcommand is supplied."""
    if ctx.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo(
            "the guided workflow requires an interactive terminal; "
            "use 'neb run --help' for the scriptable CLI",
            err=True,
        )
        raise typer.Exit(code=2)

    from neb.wizard import WizardCancelled, launch_wizard

    try:
        launch_wizard(project_root(), run_evaluation=_run)
    except WizardCancelled:
        typer.echo("\nCancelled.")


@app.command("validate")
def validate_command(
    root: Annotated[Path | None, typer.Option(help="Repository root")] = None,
) -> None:
    from neb.results import validate_repository
    from neb.tasks import get_benchmark

    base = project_root(root)
    benchmark = get_benchmark()
    evidence = validate_repository(base)
    typer.echo(
        f"valid: {len(benchmark.tasks)} native MTEB tasks, "
        f"aggregation disabled, {evidence} evidence rows"
    )


def _run(
    model: str,
    revision: str | None,
    task: list[str] | None,
    device: str,
    batch_size: int,
    dtype: str | None,
    allow_remote_code: bool,
    query_prompt: str | None,
    document_prompt: str | None,
    cache: Path,
    log_level: str,
    *,
    json_summary: bool = True,
) -> Any:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise typer.BadParameter("log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    from neb.evaluation import evaluate

    result = evaluate(
        model,
        revision,
        task,
        cache_path=cache,
        device=device,
        batch_size=batch_size,
        dtype=dtype,
        allow_remote_code=allow_remote_code,
        query_prompt=query_prompt,
        document_prompt=document_prompt,
    )
    if json_summary:
        typer.echo(
            json.dumps(
                {
                    "model_name": result.model_name,
                    "model_revision": result.model_revision,
                    "tasks": result.task_names,
                    "cache": str(cache),
                },
                indent=2,
            )
        )
    return result


@app.command("run")
def run_command(
    model: Annotated[
        str, typer.Option("--model", help="Hugging Face model ID or local model directory")
    ],
    revision: Annotated[
        str | None,
        typer.Option(
            "--revision", help="Optional exact Hub SHA; omitted resolves the current Hub HEAD"
        ),
    ] = None,
    task: Annotated[
        list[str] | None, typer.Option("--task", help="Repeat to select native task names")
    ] = None,
    device: Annotated[str, typer.Option()] = "cpu",
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    dtype: Annotated[str | None, typer.Option(help="bf16, fp16, or fp32")] = None,
    allow_remote_code: Annotated[
        bool,
        typer.Option("--allow-remote-code", help="Allow an approved exact jangedoo/* override"),
    ] = False,
    query_prompt: Annotated[str | None, typer.Option()] = None,
    document_prompt: Annotated[str | None, typer.Option()] = None,
    cache: Annotated[Path, typer.Option(help="MTEB ResultCache root")] = Path("runs"),
    log_level: Annotated[str, typer.Option(help="Python log level")] = "INFO",
) -> None:
    _run(
        model,
        revision,
        task,
        device,
        batch_size,
        dtype,
        allow_remote_code,
        query_prompt,
        document_prompt,
        cache,
        log_level,
    )


@app.command("tasks")
def tasks_command() -> None:
    from neb.tasks import get_tasks

    for task in get_tasks():
        typer.echo(
            f"{task.metadata.name}\t{task.metadata.type}\t{task.metadata.main_score}\t"
            f"{','.join(task.hf_subsets)}"
        )


@results_app.command("publish")
def results_publish(
    source: Path,
    status: Annotated[VerificationStatus, typer.Option("--status")],
    root: Annotated[Path | None, typer.Option(help="Repository root")] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace colliding scores, run settings, and complete revision metadata",
        ),
    ] = False,
) -> None:
    from neb.results import publish_results

    for path in publish_results(source.resolve(), status, project_root(root), overwrite=overwrite):
        typer.echo(path)


@app.command("export")
def export_command(
    output: Annotated[Path | None, typer.Option(help="Export directory")] = None,
    root: Annotated[Path | None, typer.Option(help="Repository root")] = None,
) -> None:
    from neb.export import export_static

    for path in export_static(project_root(root), output):
        typer.echo(path)


if __name__ == "__main__":
    app()
