"""Public library surface (spec §12)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from zarrbridge.build.builder import build
from zarrbridge.cache import fingerprint as _fp
from zarrbridge.dataset import Dataset
from zarrbridge.server.config import Config


def open(path: Path | str, config: Config | None = None) -> Dataset:  # noqa: A001
    return Dataset.open(path, config or Config.load())


def create_app(config: Config | None = None) -> Any:
    from zarrbridge.server.app import create_app as _create
    return _create(config or Config.load())


class Fingerprint:
    validate = staticmethod(_fp.validate)
    read = staticmethod(_fp.read_fingerprint)
    SCHEMA_VERSION = _fp.SCHEMA_VERSION
    DERIVATION_VERSION = _fp.DERIVATION_VERSION


__all__ = ["open", "build", "create_app", "Fingerprint", "Dataset", "Config"]
