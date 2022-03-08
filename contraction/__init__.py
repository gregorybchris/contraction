import importlib.metadata
from typing import Iterable

__path__ = __import__("pkgutil").extend_path(__path__, __name__)  # type: Iterable[str]

__version__ = importlib.metadata.version("contraction")

__all__ = [
    "__version__",
]
