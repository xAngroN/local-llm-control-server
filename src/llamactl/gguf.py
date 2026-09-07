"""Minimal, dependency-free reader for GGUF model headers.

Only the metadata needed to describe a model's geometry (and thereby
estimate its memory footprint) is extracted; tensor data is never read.
The parser walks the metadata key/value section over a memory map, so on
a multi-gigabyte file only the header pages are ever paged in.

Everything degrades to ``None`` instead of raising: a missing file, a bad
magic or an unexpected value type yields an empty/``None`` result so the
API can always answer.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path

#: GGUF metadata value type tags (see the GGUF spec).
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

#: struct formats and byte widths for the fixed-width scalar tags.
_SCALAR = {
    _UINT8: ("<B", 1),
    _INT8: ("<b", 1),
    _UINT16: ("<H", 2),
    _INT16: ("<h", 2),
    _UINT32: ("<I", 4),
    _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4),
    _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8),
    _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}

#: Approximate bytes-per-element of the common llama.cpp KV cache types.
#: The quantized types store 32-element blocks, hence the fractional cost.
KV_TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q5_1": 24.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q4_1": 20.0 / 32.0,
    "q4_0": 18.0 / 32.0,
}

#: llama.cpp ``general.file_type`` enum -> human quant label (common values).
_FILE_TYPE_LABELS = {
    0: "all_f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    7: "q8_0",
    8: "q5_0",
    9: "q5_1",
    10: "q2_k",
    11: "q3_k_s",
    12: "q3_k_m",
    13: "q3_k_l",
    14: "q4_k_s",
    15: "q4_k_m",
    16: "q5_k_s",
    17: "q5_k_m",
    18: "q6_k",
}


@dataclass(frozen=True)
class GgufInfo:
    """Geometry and size of a GGUF model, as far as it could be read."""

    architecture: str | None
    name: str | None
    quant: str | None  # human label from general.file_type
    n_layers: int | None  # <arch>.block_count
    n_embd: int | None  # <arch>.embedding_length
    n_head: int | None  # <arch>.attention.head_count
    n_head_kv: int | None  # <arch>.attention.head_count_kv
    head_dim: int | None  # key_length, else n_embd / n_head
    context_length: int | None  # <arch>.context_length (training ctx)
    param_count: int | None  # general.parameter_count, if present
    file_size_bytes: int | None

    def as_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "name": self.name,
            "quant": self.quant,
            "n_layers": self.n_layers,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_head_kv": self.n_head_kv,
            "head_dim": self.head_dim,
            "context_length": self.context_length,
            "param_count": self.param_count,
            "file_size_bytes": self.file_size_bytes,
        }


class _Cursor:
    """A byte cursor over a buffer with the GGUF primitive readers."""

    def __init__(self, buf) -> None:
        self._buf = buf
        self._pos = 0

    def read(self, fmt: str, size: int):
        value = struct.unpack_from(fmt, self._buf, self._pos)[0]
        self._pos += size
        return value

    def read_u32(self) -> int:
        return self.read("<I", 4)

    def read_u64(self) -> int:
        return self.read("<Q", 8)

    def read_string(self) -> str:
        length = self.read_u64()
        raw = self._buf[self._pos:self._pos + length]
        self._pos += length
        return bytes(raw).decode("utf-8", "replace")

    def skip_value(self, vtype: int) -> None:
        """Advance past a value of ``vtype`` without materializing it."""
        if vtype in _SCALAR:
            self._pos += _SCALAR[vtype][1]
        elif vtype == _STRING:
            self.read_string()
        elif vtype == _ARRAY:
            self._skip_array()
        else:
            raise ValueError(f"unknown gguf value type {vtype}")

    def read_scalar_value(self, vtype: int):
        """Read a scalar/string value; arrays are skipped and return ``None``."""
        if vtype in _SCALAR:
            fmt, size = _SCALAR[vtype]
            return self.read(fmt, size)
        if vtype == _STRING:
            return self.read_string()
        if vtype == _ARRAY:
            self._skip_array()
            return None
        raise ValueError(f"unknown gguf value type {vtype}")

    def _skip_array(self) -> None:
        elem_type = self.read_u32()
        count = self.read_u64()
        if elem_type in _SCALAR:
            self._pos += _SCALAR[elem_type][1] * count
        elif elem_type == _STRING:
            for _ in range(count):
                self.read_string()
        else:
            raise ValueError(f"unknown gguf array element type {elem_type}")


#: Metadata keys we care about; the rest are skipped.
def _wanted_keys(arch: str) -> dict[str, str]:
    return {
        f"{arch}.block_count": "n_layers",
        f"{arch}.embedding_length": "n_embd",
        f"{arch}.attention.head_count": "n_head",
        f"{arch}.attention.head_count_kv": "n_head_kv",
        f"{arch}.attention.key_length": "head_dim",
        f"{arch}.context_length": "context_length",
    }


def read_gguf_info(path: str | os.PathLike) -> GgufInfo | None:
    """Read a GGUF file's geometry, or ``None`` when it cannot be read.

    Two passes over the metadata are avoided: architecture is discovered
    lazily (the ``general.architecture`` key precedes the ``<arch>.*``
    keys in every real file), so a single forward scan collects the rest.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
        with p.open("rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                return _parse(mm, size)
            finally:
                mm.close()
    except (OSError, ValueError, struct.error):
        return None


def _parse(buf, file_size: int) -> GgufInfo | None:
    cur = _Cursor(buf)
    if bytes(buf[:4]) != b"GGUF":
        return None
    cur._pos = 4
    cur.read_u32()  # version
    cur.read_u64()  # tensor count
    kv_count = cur.read_u64()

    collected: dict[str, object] = {}
    arch: str | None = None
    wanted: dict[str, str] = {}
    for _ in range(kv_count):
        key = cur.read_string()
        vtype = cur.read_u32()
        if key == "general.architecture":
            arch = cur.read_scalar_value(vtype)
            if isinstance(arch, str):
                wanted = _wanted_keys(arch)
            continue
        if key in ("general.name", "general.parameter_count", "general.file_type"):
            collected[key] = cur.read_scalar_value(vtype)
            continue
        if key in wanted:
            collected[wanted[key]] = cur.read_scalar_value(vtype)
            continue
        cur.skip_value(vtype)

    n_embd = _as_int(collected.get("n_embd"))
    n_head = _as_int(collected.get("n_head"))
    head_dim = _as_int(collected.get("head_dim"))
    if head_dim is None and n_embd and n_head:
        head_dim = n_embd // n_head
    file_type = _as_int(collected.get("general.file_type"))
    return GgufInfo(
        architecture=arch,
        name=collected.get("general.name") if isinstance(collected.get("general.name"), str) else None,
        quant=_FILE_TYPE_LABELS.get(file_type) if file_type is not None else None,
        n_layers=_as_int(collected.get("n_layers")),
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=_as_int(collected.get("n_head_kv")),
        head_dim=head_dim,
        context_length=_as_int(collected.get("context_length")),
        param_count=_as_int(collected.get("general.parameter_count")),
        file_size_bytes=file_size,
    )


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


#: Bytes per (binary) gibibyte.
_GIB = 1024.0**3

#: Flat estimate for compute graph + context buffers + backend overhead.
#: Not derived from geometry (it depends on ubatch/graph internals); a
#: constant is close enough for a fit check dominated by weights + KV.
_OVERHEAD_BYTES = int(0.75 * _GIB)


def estimate_vram(
    info: GgufInfo,
    ctx_size: int,
    kv_type_k: str,
    kv_type_v: str,
) -> dict:
    """Estimate the peak VRAM for running ``info`` at ``ctx_size``.

    Returns a breakdown ``{weights, kv_cache, overhead, total}`` in bytes
    and GiB. The KV term uses **all** layers, so it is an *upper bound*:
    hybrid / sliding-window models (where only some layers keep a full KV
    cache) use less, which is why a measured peak, once known, should be
    preferred over this estimate. Any term that cannot be computed (model
    unreadable, unknown KV type) is ``None`` and excluded from the total.
    """
    weights = info.file_size_bytes
    kv_cache: int | None = None
    bk = KV_TYPE_BYTES.get(kv_type_k)
    bv = KV_TYPE_BYTES.get(kv_type_v)
    if info.n_layers and info.n_head_kv and info.head_dim and bk and bv:
        per_type = info.n_layers * info.n_head_kv * info.head_dim * ctx_size
        kv_cache = int(per_type * (bk + bv))
    parts = [p for p in (weights, kv_cache, _OVERHEAD_BYTES) if p is not None]
    total = sum(parts) if parts else None

    def gib(value: int | None) -> float | None:
        return round(value / _GIB, 3) if value is not None else None

    return {
        "weights_bytes": weights,
        "kv_cache_bytes": kv_cache,
        "overhead_bytes": _OVERHEAD_BYTES,
        "total_bytes": total,
        "weights_gib": gib(weights),
        "kv_cache_gib": gib(kv_cache),
        "overhead_gib": gib(_OVERHEAD_BYTES),
        "total_gib": gib(total),
        "approximate": True,
        "note": (
            "KV cache assumes every layer keeps a full cache (upper bound); "
            "hybrid/SWA models use less. Prefer the measured peak when known."
        ),
    }
