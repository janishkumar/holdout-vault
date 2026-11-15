"""Deterministic hashing for ledger entries, files and data frames."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CHUNK = 1 << 20


def canonical_json(obj: Any) -> bytes:
    """Serialise to compact, key-sorted UTF-8 JSON. NaN and infinity are rejected."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return text.encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def normalise_value(v: Any) -> Any:
    """Map a cell to a JSON-safe value that describes the value, not its dtype.

    Integral floats become ints (so CSV round trips hash the same), missing values become None,
    and numpy scalars become Python scalars.
    """
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        if f.is_integer() and abs(f) < 2**53:
            return int(f)
        return repr(f)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return str(v)


def row_key(values: list[Any]) -> bytes:
    return canonical_json([normalise_value(v) for v in values])


def frame_sha256(df: pd.DataFrame) -> str:
    """Hash a frame by value: column order does not matter, row order and values do."""
    cols = sorted(str(c) for c in df.columns)
    ordered = df.rename(columns=str)[cols]
    h = hashlib.sha256()
    h.update(canonical_json(cols))
    for row in ordered.itertuples(index=False, name=None):
        h.update(b"\n")
        h.update(row_key(list(row)))
    return h.hexdigest()
