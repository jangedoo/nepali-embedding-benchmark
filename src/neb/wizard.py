"""Interactive, task-first workflow for evaluating and publishing NEB evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import questionary
import typer
from questionary import Choice, Style

from neb.schemas import VerificationStatus

RunEvaluation = Callable[..., Any]

STYLE = Style(
    [
        ("qmark", "fg:#38bdf8 bold"),
        ("question", "bold"),
        ("answer", "fg:#22c55e bold"),
        ("pointer", "fg:#38bdf8 bold"),
        ("highlighted", "fg:#38bdf8 bold"),
        ("selected", "fg:#22c55e"),
        ("instruction", "fg:#94a3b8"),
        ("text", ""),
        ("disabled", "fg:#64748b italic"),
    ]
)


class WizardCancelled(Exception):
    """Raised when the user cancels an interactive prompt."""


class PromptUI(Protocol):
    def text(
        self,
        message: str,
        *,
        default: str = "",
        validate: Callable[[str], bool | str] | None = None,
        instruction: str | None = None,
    ) -> str: ...

    def select(
        self,
        message: str,
        *,
        choices: Sequence[Any],
        default: Any = None,
        instruction: str | None = None,
    ) -> Any: ...

    def checkbox(
        self,
        message: str,
        *,
        choices: Sequence[Any],
        validate: Callable[[list[Any]], bool | str] | None = None,
        instruction: str | None = None,
    ) -> list[Any]: ...

    def confirm(self, message: str, *, default: bool = True) -> bool: ...

    def write(self, message: str = "", *, style: str | None = None) -> None: ...


class QuestionaryUI:
    """Small adapter that keeps terminal prompting replaceable in tests."""

    @staticmethod
    def _answer(question: Any) -> Any:
        answer = question.ask()
        if answer is None:
            raise WizardCancelled
        return answer

    def text(
        self,
        message: str,
        *,
        default: str = "",
        validate: Callable[[str], bool | str] | None = None,
        instruction: str | None = None,
    ) -> str:
        return self._answer(
            questionary.text(
                message,
                default=default,
                validate=validate,
                instruction=instruction,
                style=STYLE,
            )
        )

    def select(
        self,
        message: str,
        *,
        choices: Sequence[Any],
        default: Any = None,
        instruction: str | None = None,
    ) -> Any:
        return self._answer(
            questionary.select(
                message,
                choices=choices,
                default=default,
                instruction=instruction,
                style=STYLE,
                use_shortcuts=True,
            )
        )

    def checkbox(
        self,
        message: str,
        *,
        choices: Sequence[Any],
        validate: Callable[[list[Any]], bool | str] | None = None,
        instruction: str | None = None,
    ) -> list[Any]:
        return self._answer(
            questionary.checkbox(
                message,
                choices=choices,
                validate=validate,
                instruction=instruction,
                style=STYLE,
            )
        )

    def confirm(self, message: str, *, default: bool = True) -> bool:
        return self._answer(questionary.confirm(message, default=default, style=STYLE))

    def write(self, message: str = "", *, style: str | None = None) -> None:
        typer.secho(message, fg=style)


@dataclass(frozen=True)
class RunConfig:
    model: str
    revision: str | None
    tasks: list[str]
    device: str
    batch_size: int
    dtype: str | None
    allow_remote_code: bool
    query_prompt: str | None
    document_prompt: str | None
    cache: Path
    log_level: str


def _required(value: str) -> bool | str:
    return True if value.strip() else "Enter a value."


def _positive_integer(value: str) -> bool | str:
    try:
        valid = int(value) > 0
    except ValueError:
        valid = False
    return True if valid else "Enter a whole number greater than zero."


def _revision(value: str) -> bool | str:
    if not value:
        return True
    valid = len(value) == 40 and all(character in "0123456789abcdef" for character in value)
    return True if valid else "Use a full lowercase 40-character commit SHA, or leave blank."


def _task_choices(tasks: Sequence[Any]) -> list[Choice]:
    choices = []
    for task in tasks:
        metadata = task.metadata
        subset_count = len(task.hf_subsets)
        subset_label = f"{subset_count} subset{'s' if subset_count != 1 else ''}"
        title = (
            f"{metadata.name}  ·  {metadata.type}  ·  {metadata.dataset['path']}  ·  {subset_label}"
        )
        choices.append(Choice(title=title, value=metadata.name))
    return choices


def _choose_device(ui: PromptUI) -> str:
    selected = ui.select(
        "Execution device",
        choices=[
            Choice("CPU", value="cpu"),
            Choice("NVIDIA CUDA", value="cuda"),
            Choice("Apple Metal (MPS)", value="mps"),
            Choice("Other device…", value="custom"),
        ],
        default="cpu",
    )
    if selected != "custom":
        return str(selected)
    return ui.text(
        "Device identifier",
        default="cuda:0",
        validate=_required,
        instruction="For example cuda:1",
    ).strip()


def collect_run_config(ui: PromptUI, tasks: Sequence[Any]) -> RunConfig:
    """Collect one explicit, task-first evaluation configuration."""
    coverage = ui.select(
        "Task coverage",
        choices=[
            Choice("Choose tasks and datasets", value="choose"),
            Choice(f"All {len(tasks)} active tasks", value="all"),
        ],
        default="choose",
    )
    if coverage == "all":
        task_names = [task.metadata.name for task in tasks]
    else:
        task_names = ui.checkbox(
            "Select tasks and datasets",
            choices=_task_choices(tasks),
            validate=lambda selected: True if selected else "Select at least one task.",
            instruction="Space selects · Enter confirms",
        )

    model = ui.text(
        "Hugging Face model ID or local model directory",
        validate=_required,
        instruction="For example intfloat/multilingual-e5-small",
    ).strip()
    is_local = Path(model).expanduser().is_dir()
    revision = None
    if not is_local:
        revision_value = ui.text(
            "Exact model revision (optional)",
            validate=_revision,
            instruction="Leave blank to resolve and record the current Hub HEAD",
        ).strip()
        revision = revision_value or None

    device = _choose_device(ui)
    batch_size = int(ui.text("Batch size", default="32", validate=_positive_integer).strip())
    dtype = ui.select(
        "Model precision",
        choices=[
            Choice("Model default", value=None),
            Choice("bfloat16", value="bf16"),
            Choice("float16", value="fp16"),
            Choice("float32", value="fp32"),
        ],
        default=None,
    )
    cache = Path(
        ui.text(
            "Result cache directory",
            default="runs",
            validate=_required,
            instruction="Existing task results resume automatically",
        ).strip()
    ).expanduser()

    query_prompt = None
    document_prompt = None
    allow_remote_code = False
    log_level = "INFO"
    if ui.confirm("Configure advanced options?", default=False):
        query_value = ui.text(
            "Query prompt override (optional)",
            instruction="Leave blank to use model-native prompts",
        ).strip()
        document_value = ui.text(
            "Document prompt override (optional)",
            instruction="Leave blank to use model-native prompts",
        ).strip()
        query_prompt = query_value or None
        document_prompt = document_value or None
        if not is_local:
            allow_remote_code = ui.confirm(
                "Allow remote code for an approved, exactly pinned jangedoo/* override?",
                default=False,
            )
        log_level = ui.select(
            "Log level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
        )

    return RunConfig(
        model=model,
        revision=revision,
        tasks=list(task_names),
        device=device,
        batch_size=batch_size,
        dtype=dtype,
        allow_remote_code=allow_remote_code,
        query_prompt=query_prompt,
        document_prompt=document_prompt,
        cache=cache,
        log_level=log_level,
    )


def _show_review(ui: PromptUI, config: RunConfig) -> None:
    ui.write("\nEvaluation plan", style="cyan")
    ui.write(f"  Model      {config.model}")
    ui.write(f"  Revision   {config.revision or 'resolve current Hub HEAD'}")
    ui.write(f"  Tasks      {len(config.tasks)} selected")
    for task in config.tasks:
        ui.write(f"             • {task}")
    ui.write(
        f"  Runtime    {config.device} · batch {config.batch_size} · {config.dtype or 'default'}"
    )
    ui.write(f"  Cache      {config.cache}")
    ui.write(
        "  Prompts    "
        + ("custom overrides" if config.query_prompt or config.document_prompt else "model-native")
    )
    remote_code = "allowed under NEB policy" if config.allow_remote_code else "disabled"
    ui.write(f"  Remote code {remote_code}")


def _is_checkout(root: Path) -> bool:
    return (root / "pyproject.toml").is_file() and (root / "src/neb").is_dir()


def _cache_source(cache: Path, result: Any) -> Path:
    model_dir = result.model_name.replace("/", "__").replace(" ", "_")
    return cache.resolve() / "results" / model_dir / result.model_revision


def _publish_and_export(ui: PromptUI, root: Path, config: RunConfig, result: Any) -> None:
    local_name = str(result.model_name).startswith("local/")
    local_revision = str(result.model_revision).startswith("local-")
    if local_name or local_revision:
        ui.write(
            "\nLocal model results remain in the cache and cannot be published.",
            style="yellow",
        )
        return
    if not _is_checkout(root):
        ui.write(
            "\nPublishing and dashboard export require running neb inside an NEB checkout.",
            style="yellow",
        )
        return

    publish_status = ui.select(
        "Publish this model revision as canonical evidence?",
        choices=[
            Choice("Not now", value=None),
            Choice("Community · schema-checked, explicitly unverified", value="community"),
            Choice("Verified · maintainers only", value="verified"),
        ],
        default=None,
    )
    published = False
    if publish_status is not None:
        status = VerificationStatus(publish_status)
        if status is VerificationStatus.verified and not ui.confirm(
            "Confirm this run was produced in the maintainer evaluation environment",
            default=False,
        ):
            ui.write("Publication skipped; the cache is unchanged.", style="yellow")
        else:
            overwrite = ui.confirm(
                "Replace colliding scores and settings if they differ?",
                default=False,
            )
            from neb.results import publish_results

            source = _cache_source(config.cache, result)
            paths = publish_results(source, status, root, overwrite=overwrite)
            ui.write(
                f"\nPublished {len(paths)} task result file(s) as {status.value}.",
                style="green",
            )
            published = True

    if ui.confirm(
        "Regenerate the static dashboard export?",
        default=published,
    ):
        from neb.export import export_static

        paths = export_static(root)
        ui.write(f"Exported {len(paths)} dashboard data file(s).", style="green")


def launch_wizard(
    root: Path,
    *,
    run_evaluation: RunEvaluation,
    ui: PromptUI | None = None,
) -> Any | None:
    """Guide the user through an evaluation and its optional evidence lifecycle."""
    from neb.tasks import get_tasks

    prompt = ui or QuestionaryUI()
    prompt.write("\nNEB · Nepali Embedding Benchmark", style="cyan")
    prompt.write("Task-first evaluation with native MTEB evidence. No overall score.\n")
    config = collect_run_config(prompt, get_tasks())
    _show_review(prompt, config)
    if not prompt.confirm("Start evaluation?", default=True):
        prompt.write("Evaluation not started; no files were changed.", style="yellow")
        return None

    prompt.write("\nStarting evaluation…", style="cyan")
    result = run_evaluation(
        config.model,
        config.revision,
        config.tasks,
        config.device,
        config.batch_size,
        config.dtype,
        config.allow_remote_code,
        config.query_prompt,
        config.document_prompt,
        config.cache,
        config.log_level,
        json_summary=False,
    )
    prompt.write("\nEvaluation complete.", style="green")
    prompt.write(f"  Model      {result.model_name}@{result.model_revision}")
    prompt.write(f"  Tasks      {len(result.task_names)} completed or resumed")
    prompt.write(f"  Cache      {config.cache}")
    _publish_and_export(prompt, root, config, result)
    return result
