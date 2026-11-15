"""Provenance stamp for the code that produced a submission."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from holdout_vault.core.hashing import file_sha256

LOCKFILES = ("uv.lock", "poetry.lock", "requirements.lock", "requirements.txt")


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout


def collect_stamp(repo_dir: str | Path | None, seed: int | None = None) -> dict[str, Any]:
    stamp: dict[str, Any] = {
        "git_commit": None,
        "git_dirty": None,
        "lockfile": None,
        "lockfile_sha256": None,
        "seed": seed,
    }
    if repo_dir is None:
        return stamp
    repo = Path(repo_dir)
    head = _git(repo, "rev-parse", "HEAD")
    if head:
        stamp["git_commit"] = head.strip()
        status = _git(repo, "status", "--porcelain")
        stamp["git_dirty"] = bool(status and status.strip())
    for name in LOCKFILES:
        p = repo / name
        if p.is_file():
            stamp["lockfile"] = name
            stamp["lockfile_sha256"] = file_sha256(p)
            break
    return stamp
