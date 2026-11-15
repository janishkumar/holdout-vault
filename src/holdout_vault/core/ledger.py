"""Append-only, hash-chained JSONL ledger.

Each line is one entry: ``seq``, ``ts``, ``kind``, ``payload``, ``prev_hash`` and ``hash``, where
``hash`` is SHA-256 over the canonical JSON of the entry without ``hash``. Editing, deleting,
reordering or truncating any entry makes :meth:`Ledger.verify` fail at the first bad sequence number.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from holdout_vault.core.hashing import canonical_json, sha256_hex

GENESIS_HASH = "0" * 64

KINDS = frozenset({"register", "score", "evaluate", "refused", "grant", "budget", "tamper", "unseal"})


@dataclass(frozen=True)
class Entry:
    seq: int
    ts: str
    kind: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    count: int
    bad_seq: int | None = None
    reason: str | None = None
    head_hash: str | None = None


def entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return sha256_hex(canonical_json(body))


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read_raw(self) -> list[str]:
        if not self.path.exists():
            return []
        return [ln for ln in self.path.read_text(encoding="utf-8").split("\n") if ln.strip()]

    def entries(self) -> list[Entry]:
        return [Entry(**json.loads(ln)) for ln in self._read_raw()]

    def head(self) -> Entry | None:
        lines = self._read_raw()
        return Entry(**json.loads(lines[-1])) if lines else None

    def verify(self) -> VerifyResult:
        prev = GENESIS_HASH
        count = 0
        for i, line in enumerate(self._read_raw()):
            try:
                obj = json.loads(line)
                Entry(**obj)
            except (json.JSONDecodeError, TypeError) as exc:
                return VerifyResult(False, count, i, f"cannot parse entry {i}: {exc}")
            if obj["seq"] != i:
                return VerifyResult(False, count, i, f"expected seq {i}, found {obj['seq']}")
            if obj["prev_hash"] != prev:
                return VerifyResult(False, count, i, f"entry {i} does not link to the previous entry")
            if entry_hash(obj) != obj["hash"]:
                return VerifyResult(False, count, i, f"entry {i} content does not match its hash")
            prev = obj["hash"]
            count += 1
        return VerifyResult(True, count, head_hash=None if count == 0 else prev)

    def append(self, kind: str, payload: dict[str, Any]) -> Entry:
        if kind not in KINDS:
            raise ValueError(f"unknown ledger kind {kind!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                check = self.verify()
                if not check.ok:
                    raise RuntimeError(f"ledger is broken, refusing to append: {check.reason}")
                body = {
                    "seq": check.count,
                    "ts": datetime.now(UTC).isoformat(timespec="microseconds"),
                    "kind": kind,
                    "payload": payload,
                    "prev_hash": check.head_hash or GENESIS_HASH,
                }
                body["hash"] = entry_hash(body)
                f.write(canonical_json(body).decode() + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return Entry(**body)
