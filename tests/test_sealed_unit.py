"""Sealed-mode plumbing that does not need a Docker daemon."""

import pytest

from holdout_vault import sealed
from holdout_vault.cli import main
from holdout_vault.core.errors import VaultError


def test_build_context_has_package_dockerfile_and_compose(tmp_path):
    ctx = sealed.build_context(tmp_path)
    assert (ctx / "Dockerfile").is_file() and (ctx / "compose.yaml").is_file()
    assert (ctx / "src" / "holdout_vault" / "core" / "vault.py").is_file()
    assert not list(ctx.rglob("__pycache__"))
    pyproject = (ctx / "pyproject.toml").read_text()
    assert 'holdout-vault = "holdout_vault.cli:main"' in pyproject
    assert "scikit-learn" in pyproject


def test_compose_file_keeps_api_on_loopback_and_drops_privileges(tmp_path):
    compose = (sealed.build_context(tmp_path) / "compose.yaml").read_text()
    assert '"127.0.0.1:${HV_PORT:-7341}:7341"' in compose
    assert "read_only: true" in compose and "cap_drop: [ALL]" in compose


def test_compose_command_uses_project_and_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOLDOUT_VAULT_COMPOSE_PROJECT", "hv-x")
    cmd = sealed.compose(tmp_path, "ps")
    assert cmd[:4] == ["docker", "compose", "-p", "hv-x"]
    assert cmd[-1] == "ps" and cmd[5].endswith("docker-context/compose.yaml")


def test_port_from_env(monkeypatch):
    monkeypatch.setenv("HOLDOUT_VAULT_PORT", "7350")
    assert sealed.port() == 7350


def test_wait_healthy_times_out():
    with pytest.raises(VaultError, match="healthy"):
        sealed.wait_healthy("http://127.0.0.1:9", timeout=0.5)


def test_run_raises_with_stderr():
    with pytest.raises(VaultError, match="failed"):
        sealed._run(["python3", "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"])


def test_sealed_admin_commands_forward_to_container(monkeypatch, home):
    calls = []
    monkeypatch.setattr(sealed, "exec_cli", lambda h, argv: calls.append(argv) or 0)
    assert main(["grant", "--sealed", "--project", "p", "2", "--note", "n", "--by", "me"]) == 0
    assert main(["budget", "--sealed", "--project", "p", "5", "--by", "me"]) == 0
    assert main(["status", "--sealed", "--project", "p", "--json"]) == 0
    assert main(["verify-ledger", "--sealed", "--project", "p"]) == 0
    assert main(["ledger", "--sealed", "--project", "p", "--tail", "3"]) == 0
    assert calls == [
        ["grant", "--project", "p", "2", "--by", "me", "--note", "n"],
        ["budget", "--project", "p", "5", "--by", "me"],
        ["status", "--project", "p", "--json"],
        ["verify-ledger", "--project", "p"],
        ["ledger", "--project", "p", "--tail", "3"],
    ]


def test_sealed_register_builds_container_arguments(monkeypatch, home, split_files, tmp_path):
    seen = {}

    def fake_register(h, args, files, export):
        seen.update(args=args, files=files, export=export)
        return 0

    monkeypatch.setattr(sealed, "register", fake_register)
    train, test = split_files
    code = main(
        ["register", "--sealed", "--project", "p", "--name", "d", "--train", str(train), "--test", str(test),
         "--target", "churn", "--metrics", "accuracy", "--export-dir", str(tmp_path / "ws")]
    )  # fmt: skip
    assert code == 0
    assert seen["args"][:4] == ["--project", "p", "--name", "d"]
    assert "--agent-export-dir" in seen["args"]
    assert set(seen["files"]) == {"--train", "--test"}
