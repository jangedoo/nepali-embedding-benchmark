"""Public API for the Nepali Embedding Benchmark."""

from importlib.metadata import PackageNotFoundError, version

from neb.api import evaluate, get_benchmark, get_tasks, resolve_model

try:
    __version__ = version("nepali-embedding-benchmark")
except PackageNotFoundError:  # source checkout
    __version__ = "0.3.0"

__all__ = [
    "__version__",
    "evaluate",
    "get_benchmark",
    "get_tasks",
    "resolve_model",
]
