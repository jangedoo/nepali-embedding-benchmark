"""MTEB-native public API."""

from neb.evaluation import evaluate
from neb.models import resolve_model
from neb.tasks import get_benchmark, get_tasks

__all__ = ["evaluate", "get_benchmark", "get_tasks", "resolve_model"]
