from holdout_vault.core.provenance import collect_stamp


def test_clean_repo(git_repo):
    s = collect_stamp(git_repo, seed=42)
    assert len(s["git_commit"]) == 40
    assert s["git_dirty"] is False
    assert s["lockfile"] == "uv.lock"
    assert len(s["lockfile_sha256"]) == 64
    assert s["seed"] == 42


def test_dirty_repo(git_repo):
    (git_repo / "train.py").write_text("print('changed')\n")
    assert collect_stamp(git_repo)["git_dirty"] is True


def test_untracked_file_makes_dirty(git_repo):
    (git_repo / "new.py").write_text("x = 1\n")
    assert collect_stamp(git_repo)["git_dirty"] is True


def test_not_a_repo(tmp_path):
    s = collect_stamp(tmp_path)
    assert s["git_commit"] is None and s["git_dirty"] is None and s["lockfile"] is None


def test_none_repo():
    assert collect_stamp(None, seed=1) == {
        "git_commit": None,
        "git_dirty": None,
        "lockfile": None,
        "lockfile_sha256": None,
        "seed": 1,
    }
