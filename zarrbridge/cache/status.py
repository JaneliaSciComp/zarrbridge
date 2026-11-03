"""`zarrbridge status` rows (spec §9.6)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from zarrbridge.cache.layout import BESIDE_DIR
from zarrbridge.dataset import Dataset
from zarrbridge.readers.base import ReaderRejected
from zarrbridge.readers.registry import sniffable


def iter_files(target: Path) -> Iterator[Path]:
    target = Path(target)
    if target.is_file():
        yield target
        return
    for p in sorted(target.rglob("*")):
        if BESIDE_DIR in p.parts or p.name.startswith("."):
            continue
        if p.is_file() and sniffable(p):
            yield p


def dataset_status_row(path: Path, config: Any) -> dict:
    try:
        ds = Dataset.open(path, config)
    except ReaderRejected as e:
        return {"path": str(path), "reader": "-", "classification": "unservable", "levels": "unservable",
                "cache_bytes": 0, "fingerprint": "-", "mismatch": str(e)}
    try:
        s = ds.status()
        return {"path": str(path), "reader": s["reader"], "classification": s["classification"],
                "levels": s["levels"]["state"], "cache_bytes": s["cache"]["bytes"],
                "fingerprint": s["cache"]["fingerprint"], "mismatch": s["cache"]["mismatch"] or ""}
    finally:
        ds.close()
