"""Run an agent's model away from the labels.

Sealed mode starts a throwaway runner container: no network, read-only root filesystem, no vault
volume, unprivileged user, CPU/memory/pid/time limits. The model and the test features are mounted
read-only and only a predictions file comes back. Watch-only mode runs the same harness in a
subprocess, which isolates nothing and is labelled accordingly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_IMAGE = "holdout-vault:local"
Progress = Callable[[str], None]


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunnerConfig:
    use_docker: bool = False
    image: str = DEFAULT_IMAGE
    memory: str = "2g"
    cpus: str = "2"
    pids: int = 256
    timeout_s: int = 900


def docker_command(model: Path, workdir: Path, id_col: str, cfg: RunnerConfig) -> list[str]:
    target = "/model" if model.is_dir() else f"/model/{model.name}"
    mount_src = model if model.is_dir() else model.parent
    return [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=256m",
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
        "--user", "65534:65534",
        "--memory", cfg.memory,
        "--cpus", cfg.cpus,
        "--pids-limit", str(cfg.pids),
        "-v", f"{mount_src.resolve()}:/model:ro",
        "-v", f"{(workdir / 'features.csv').resolve()}:/data/features.csv:ro",
        "-v", f"{(workdir / 'out').resolve()}:/out:rw",
        "--entrypoint", "python",
        cfg.image,
        "-m", "holdout_vault.runner.harness",
        "--model", target,
        "--features", "/data/features.csv",
        "--id-col", id_col,
        "--out", "/out/predictions.csv",
    ]  # fmt: skip


def subprocess_command(model: Path, workdir: Path, id_col: str) -> list[str]:
    return [
        sys.executable, "-m", "holdout_vault.runner.harness",
        "--model", str(model.resolve()),
        "--features", str(workdir / "features.csv"),
        "--id-col", id_col,
        "--out", str(workdir / "out" / "predictions.csv"),
    ]  # fmt: skip


def run_model(
    model_path: str | Path,
    features: pd.DataFrame,
    id_col: str,
    cfg: RunnerConfig,
    progress: Progress | None = None,
) -> pd.DataFrame:
    model = Path(model_path)
    if not model.exists():
        raise RunnerError(f"model not found: {model}")
    say = progress or (lambda _msg: None)
    workdir = Path(tempfile.mkdtemp(prefix="hv-run-"))
    try:
        (workdir / "out").mkdir()
        os.chmod(workdir / "out", 0o777)  # the container runs as an unprivileged user
        features.to_csv(workdir / "features.csv", index=False)
        os.chmod(workdir / "features.csv", 0o644)
        if cfg.use_docker:
            say("starting isolated runner container")
            cmd = docker_command(model, workdir, id_col, cfg)
        else:
            say("running model in a local subprocess (watch-only mode, not isolated)")
            cmd = subprocess_command(model, workdir, id_col)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(f"model run exceeded {cfg.timeout_s}s") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            raise RunnerError("model run failed: " + " | ".join(tail))
        say("predictions written")
        return pd.read_csv(workdir / "out" / "predictions.csv")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
