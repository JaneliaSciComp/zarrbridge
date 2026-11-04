"""Configuration: TOML file, ZB_* environment, CLI overrides (CLI > env > file > defaults)."""
from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_type_hints

_UNITS = {"": 1, "b": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12,
          "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
          "ki": 2**10, "mi": 2**20, "gi": 2**30, "ti": 2**40,
          "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}


def parse_bytes(s: str | int) -> int:
    if isinstance(s, int):
        return s
    m = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]*)\s*", str(s))
    if not m or m.group(2).lower() not in _UNITS:
        raise ValueError(f"bad byte size: {s!r}")
    return int(float(m.group(1)) * _UNITS[m.group(2).lower()])


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8800
    uds: str = ""
    token: str = ""
    cors_origins: list[str] = field(default_factory=list)
    browse: bool = True
    trusted_path_header: str = ""
    allow_prefixes: list[str] = field(default_factory=lambda: ["/groups/**", "/nrs/**"])
    immutable_chunks: bool = False
    insecure: bool = False
    follow_symlinks: bool = False
    log_format: str = "text"
    index_html: str = ""  # path of an HTML file to serve at / instead of the root listing
    ssl_certfile: str = ""  # HTTPS (needed when viewers are served over https, which blocks mixed content)
    ssl_keyfile: str = ""
    viewer_dir: str = ""  # a standalone Neuroglancer client build (dist/client) to serve at /neuroglancer/


@dataclass
class CacheConfig:
    mode: str = "beside"  # beside | central
    central_root: str = "~/.cache/zarrbridge/pyramids"
    recheck_seconds: float = 30.0


@dataclass
class BuildConfig:
    auto: bool = True
    auto_max_bytes: int = 20 * 2**30
    slots: int = 1
    nice: int = 10
    factor: str = "auto"
    min_axis: int = 64
    max_levels: int = 8
    codec: str = "zstd:3"
    method: str = "mean"
    lock_stale_seconds: float = 6 * 3600
    factor_by_glob: dict[str, str] = field(default_factory=dict)


@dataclass
class MrcReaderConfig:
    chunk_mode: str = "cubes"
    chunk_shape: tuple[int, int, int] = (64, 64, 64)
    stack_globs: list[str] = field(default_factory=lambda: ["*.st", "*.ali", "*_tilt.mrc", "*/TiltSeries/*", "*/Gains/*"])
    volume_globs: list[str] = field(default_factory=lambda: ["*/Tomograms/*", "*_ctf.mrc"])
    assume_mode0: str | None = None


@dataclass
class TiffReaderConfig:
    prefer: str = "refkv"  # refkv | tifffile


@dataclass
class RefkvReaderConfig:
    connections: int = 8
    max_reads: int = 64


@dataclass
class ReadersConfig:
    mrc: MrcReaderConfig = field(default_factory=MrcReaderConfig)
    tiff: TiffReaderConfig = field(default_factory=TiffReaderConfig)
    refkv: RefkvReaderConfig = field(default_factory=RefkvReaderConfig)


@dataclass
class TranscodeConfig:
    # dtypes viewers cannot render -> what to serve instead (Neuroglancer has no float64/float16/int64)
    dtype_map: dict[str, str] = field(default_factory=lambda: {"float64": "float32", "float16": "float32"})


@dataclass
class LimitsConfig:
    max_open_datasets: int = 64
    chunk_cache_bytes: int = 256 * 2**20
    max_decodes: int = 0  # 0 -> os.cpu_count()
    max_reads_per_chunk: int = 256
    max_chunk_bytes: int = 64 * 2**20


_BYTE_FIELDS = {("build", "auto_max_bytes"), ("limits", "chunk_cache_bytes"), ("limits", "max_chunk_bytes")}


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    roots: dict[str, str] = field(default_factory=dict)
    cache: CacheConfig = field(default_factory=CacheConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    readers: ReadersConfig = field(default_factory=ReadersConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    transcode: TranscodeConfig = field(default_factory=TranscodeConfig)

    @classmethod
    def load(cls, file: Path | str | None = None, env: Mapping[str, str] | None = None,
             overrides: Mapping[str, Any] | None = None) -> "Config":
        env = os.environ if env is None else env
        data: dict[str, Any] = {}
        if file is not None and Path(file).exists():
            with open(file, "rb") as f:
                _merge(data, tomllib.load(f))
        _merge(data, _from_env(env))
        if overrides:
            _merge(data, overrides)
        return _build(cls, data, ())

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if d["server"].get("token"):
            d["server"]["token"] = "***"
        return d

    def max_decodes(self) -> int:
        return self.limits.max_decodes or (os.cpu_count() or 1)


def _merge(dst: dict, src: Mapping) -> None:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = dict(v) if isinstance(v, Mapping) else v


def _from_env(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in env.items():
        if not key.startswith("ZB_"):
            continue
        parts = key[3:].lower().split("_", 1)
        if parts[0] == "roots":
            out["roots"] = {k.strip(): v.strip() for k, v in (p.split("=", 1) for p in val.split(",") if "=" in p)}
        elif parts[0] == "readers" and len(parts) == 2:
            sub, _, fld = parts[1].partition("_")
            out.setdefault("readers", {}).setdefault(sub, {})[fld] = val
        elif len(parts) == 2:
            out.setdefault(parts[0], {})[parts[1]] = val
    return out


def _build(cls: type, data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    kwargs: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        ftype = hints.get(f.name)
        ftype = ftype if isinstance(ftype, type) else None
        default = f.default if f.default is not dataclasses.MISSING else (f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
        if ftype is not None and is_dataclass(ftype):
            kwargs[f.name] = _build(ftype, v, path + (f.name,))
        else:
            kwargs[f.name] = _coerce(v, default, path + (f.name,))
    return cls(**kwargs)


def _coerce(v: Any, default: Any, path: tuple[str, ...]) -> Any:
    if path in _BYTE_FIELDS:
        return parse_bytes(v)
    if isinstance(default, bool):
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    if isinstance(default, tuple):
        if isinstance(v, str):
            v = [x for x in re.split(r"[,\s]+", v.strip("()[]")) if x]
        return tuple(int(x) for x in v)
    if isinstance(default, list):
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        return list(v)
    if isinstance(default, dict):
        return dict(v)
    return v
