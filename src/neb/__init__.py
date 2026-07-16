"""Public API for the Nepali Embedding Benchmark."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nepali-embedding-benchmark")
except PackageNotFoundError:  # source checkout
    __version__ = "0.3.0"

__all__ = [
    "__version__",
]
