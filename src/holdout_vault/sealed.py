"""Sealed mode operations: build and start the vault container, and run human commands inside it."""

from __future__ import annotations

import getpass
import json
import os
import shutil
import subprocess
import time
import urllib.request
import uuid
from importlib import resources
from pathlib import Path

from holdout_vault import __version__
from holdout_vault.core.errors import VaultError
from holdout_vault.core.store import default_home
from holdout_vault.remote import SEALED_CONFIG

DEPENDENCIES = ["mcp>=2.3,<3", "numpy>=1.26", "pandas>=2.2", "pyarrow>=16", "scikit-learn>=1.5", "scipy>=1.13"]


def compose_project() -> str:
    return os.environ.get("HOLDOUT_VAULT_COMPOSE_PROJECT", "holdout-vault")


def port() -> int:
    return int(os.environ.get("HOLDOUT_VAULT_PORT", "7341"))


def context_dir(home: Path) -> Path:
    return home / "docker-context"


def build_context(home: Path) -> Path:
    """Assemble a build context from the installed package, so `uvx` installs can build the image too."""
    ctx = context_dir(home)
    if ctx.exists():
        shutil.rmtree(ctx)
    pkg = Path(str(resources.files("holdout_vault")))
    shutil.copytree(pkg, ctx / "src" / "holdout_vault", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    deps = ",\n".join(f'  "{d}"' for d in DEPENDENCIES)
    (ctx / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "holdout-vault"\nversion = "{__version__}"\nrequires-python = ">=3.12"\n'
        f"dependencies = [\n{deps}\n]\n"
        '[project.scripts]\nholdout-vault = "holdout_vault.cli:main"\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/holdout_vault"]\n'
    )
    shutil.copy(pkg / "docker" / "Dockerfile", ctx / "Dockerfile")
    shutil.copy(pkg / "docker" / "compose.yaml", ctx / "compose.yaml")
    return ctx


def compose(home: Path, *args: str) -> list[str]:
    return ["docker", "compose", "-p", compose_project(), "-f", str(context_dir(home) / "compose.yaml"), *args]


def _run(cmd: list[str], *, check: bool = True, capture: bool = True, env: dict | None = None):
    proc = subprocess.run(cmd, capture_output=capture, text=True, env={**os.environ, **(env or {})})
    if check and proc.returncode != 0:
        raise VaultError(f"`{' '.join(cmd[:4])} ...` failed: {(proc.stderr or proc.stdout).strip()[-500:]}")
    return proc


def wait_healthy(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise VaultError(f"the vault did not become healthy at {url}")


def up(home: str | Path | None = None, build: bool = True) -> dict:
    home = Path(home) if home else default_home()
    home.mkdir(parents=True, exist_ok=True)
    build_context(home)
    env = {"HV_PORT": str(port())}
    _run(compose(home, "up", "-d", *(["--build"] if build else [])), env=env)
    url = f"http://127.0.0.1:{port()}"
    wait_healthy(url)
    token = _run(compose(home, "exec", "-T", "vault", "cat", "/vault/api_token"), env=env).stdout.strip()
    if not token:
        raise VaultError("could not read the vault API token")
    cfg = {"url": url, "token": token, "compose_project": compose_project()}
    path = home / SEALED_CONFIG
    path.write_text(json.dumps(cfg, indent=2))
    path.chmod(0o600)
    return cfg


def down(home: str | Path | None = None, purge: bool = False) -> None:
    home = Path(home) if home else default_home()
    _run(compose(home, "down", *(["-v"] if purge else [])), env={"HV_PORT": str(port())})
    (home / SEALED_CONFIG).unlink(missing_ok=True)


def exec_cli(home: str | Path | None, args: list[str]) -> int:
    """Run a human CLI command inside the vault container, streaming its output."""
    home = Path(home) if home else default_home()
    cmd = compose(home, "exec", "-T", "vault", "holdout-vault", *args)
    return subprocess.run(cmd, env={**os.environ, "HV_PORT": str(port())}).returncode


def container_id(home: Path) -> str:
    cid = _run(compose(home, "ps", "-q", "vault"), env={"HV_PORT": str(port())}).stdout.strip()
    if not cid:
        raise VaultError("the vault container is not running; run `holdout-vault up` first")
    return cid


def register(home: str | Path | None, args: list[str], files: dict[str, Path], export_dir: Path) -> int:
    """Copy split files into the container, register there, copy the agent's exports back, clean up.

    ``args`` are register arguments without file or export options; ``files`` maps option names
    (``--train``, ``--test``, ``--scorer``) to host paths.
    """
    home = Path(home) if home else default_home()
    cid = container_id(home)
    job = uuid.uuid4().hex[:12]
    inbox, outbox = f"/scratch/in/{job}", f"/scratch/export/{job}"
    env = {"HV_PORT": str(port())}
    _run(compose(home, "exec", "-T", "vault", "mkdir", "-p", inbox), env=env)
    remote_args: list[str] = []
    try:
        for opt, path in files.items():
            dest = f"{inbox}/{opt.strip('-')}{Path(path).suffix}"
            _run(["docker", "cp", str(Path(path).resolve()), f"{cid}:{dest}"])
            remote_args += [opt, dest]
        code = exec_cli(home, ["register", *args, *remote_args, "--export-dir", outbox, "--by", getpass.getuser()])
        if code != 0:
            return code
        export_dir.mkdir(parents=True, exist_ok=True)
        name = args[args.index("--name") + 1]
        _run(["docker", "cp", f"{cid}:{outbox}/{name}", str(export_dir)])
        return 0
    finally:
        _run(compose(home, "exec", "-T", "-u", "root", "vault", "rm", "-rf", inbox, outbox), check=False, env=env)
