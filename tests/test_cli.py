import json

from holdout_vault.cli import main


def _register(split_files, tmp_path, *extra):
    train, test = split_files
    return main(
        [
            "register",
            "--project",
            "demo",
            "--name",
            "churn",
            "--train",
            str(train),
            "--test",
            str(test),
            "--target",
            "churn",
            "--id-col",
            "customer_id",
            "--metrics",
            "accuracy,roc_auc",
            "--export-dir",
            str(tmp_path / "ws"),
            *extra,
        ]
    )


def test_register_and_status(home, split_files, tmp_path, capsys):
    assert _register(split_files, tmp_path) == 0
    out = capsys.readouterr().out
    assert "churn" in out and "test_features.csv" in out
    assert main(["status", "--project", "demo", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["remaining"] == 20 and status["datasets"][0]["name"] == "churn"


def test_register_error_exit_code(home, split_files, tmp_path, capsys):
    _register(split_files, tmp_path)
    assert _register(split_files, tmp_path) == 2
    assert "already registered" in capsys.readouterr().err


def test_budget_and_grant(home, split_files, tmp_path, capsys):
    _register(split_files, tmp_path)
    assert main(["budget", "--project", "demo", "5"]) == 0
    assert main(["grant", "--project", "demo", "2", "--note", "final check"]) == 0
    capsys.readouterr()
    main(["status", "--project", "demo", "--json"])
    assert json.loads(capsys.readouterr().out)["remaining"] == 7


def test_verify_ledger_ok_and_broken(home, split_files, tmp_path, capsys):
    _register(split_files, tmp_path)
    assert main(["verify-ledger", "--project", "demo"]) == 0
    assert "OK" in capsys.readouterr().out
    ledger = home / "projects" / "demo" / "ledger.jsonl"
    ledger.write_text(ledger.read_text().replace('"n_test":200', '"n_test":201'))
    assert main(["verify-ledger", "--project", "demo"]) == 1
    assert "BROKEN" in capsys.readouterr().out


def test_ledger_tail(home, split_files, tmp_path, capsys):
    _register(split_files, tmp_path)
    main(["grant", "--project", "demo", "1"])
    capsys.readouterr()
    assert main(["ledger", "--project", "demo", "--tail", "1"]) == 0
    out = capsys.readouterr().out
    assert "grant" in out and "register" not in out


def test_no_command_prints_help(capsys):
    assert main([]) == 2
