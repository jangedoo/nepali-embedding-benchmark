"""Public API for the Nepali Embedding Benchmark."""

from importlib.metadata import PackageNotFoundError, version

from neb.api import Benchmark, evaluate, get_benchmark, get_models, get_tasks

try:
    __version__ = version("nepali-embedding-benchmark")
except PackageNotFoundError:  # source checkout
    __version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "__version__",
    "evaluate",
    "get_benchmark",
    "get_models",
    "get_tasks",
]
