"""Reader selection (spec §5.5)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from zarrbridge.readers.base import Opened, ReaderRejected
from zarrbridge.readers.mrc.header import MRC_SUFFIXES, sniff_mrc
from zarrbridge.readers.mrc.reader import MrcReader
from zarrbridge.readers.refkv_reader import RefkvReader
from zarrbridge.readers.tiff.reader import TIFF_MAGICS, TiffReader

HEAD_BYTES = 64 * 1024
_mrc = MrcReader()
_tiff = TiffReader()
_refkv = RefkvReader()


def read_head(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read(HEAD_BYTES)


def sniffable(path: Path) -> bool:
    """Cheap test used by the browse page and `build <dir>`."""
    try:
        if not path.is_file():
            return False
        head = read_head(path)
    except OSError:
        return False
    return (path.suffix.lower() in MRC_SUFFIXES and sniff_mrc(head)) or head[:4] in TIFF_MAGICS or RefkvReader.sniff(path, head)


def open_dataset_reader(path: Path, config: Any, index_path: Path | None = None) -> Opened:
    """Pick a reader per §5.5 and open; `reader_reason` says why. Raises ReaderRejected."""
    path = Path(path)
    if not path.is_file():
        raise ReaderRejected("not a regular file")
    head = read_head(path)
    is_mrc = path.suffix.lower() in MRC_SUFFIXES and sniff_mrc(head)
    if is_mrc and config.readers.mrc.chunk_mode != "native":
        op = _mrc.open(path, config)
        op.reader_reason = "mrc magic and suffix"
        return op
    if head[:4] in TIFF_MAGICS:
        order = ("refkv", "tiff") if config.readers.tiff.prefer == "refkv" else ("tiff", "refkv")
        errors = []
        for name in order:
            try:
                if name == "refkv":
                    op = _refkv.open(path, config, index_path=index_path)
                else:
                    op = _tiff.open(path, config)
            except ReaderRejected as e:
                errors.append(str(e))
                continue
            op.reader_reason = (f"{name}: {op.reader_reason or 'ok'}" if not errors else f"{'; '.join(errors)} -> {name}")
            return op
        raise ReaderRejected("; ".join(errors))
    if RefkvReader.sniff(path, head):
        op = _refkv.open(path, config, index_path=index_path)
        op.reader_reason = f"refkv sniff: {op.identity.extras.get('format')}"
        return op
    raise ReaderRejected("no reader")
