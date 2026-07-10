"""Road Damage Detection package.

Kept intentionally light: importing the package pulls in only the config
loader, not torch/ultralytics, so tooling and tests stay fast.
"""

from .config import Config

__version__ = "0.1.0"
__all__ = ["Config", "__version__"]
