"""zarrbridge: virtual OME-Zarr 0.5 serving for legacy image files."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("zarrbridge")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

from zarrbridge.api import Config, Dataset, Fingerprint, build, create_app, open  # noqa: E402,F401

__all__ = ["__version__", "open", "build", "create_app", "Fingerprint", "Dataset", "Config"]
