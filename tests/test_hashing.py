import math

import pandas as pd

from holdout_vault.core.hashing import canonical_json, file_sha256, frame_sha256, sha256_hex


def test_canonical_json_is_key_order_invariant():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})


def test_canonical_json_rejects_nan():
    try:
        canonical_json({"x": math.nan})
    except ValueError:
        return
    raise AssertionError("NaN must not be serialisable")


def test_sha256_hex_known_value():
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_file_sha256(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"")
    assert file_sha256(p) == sha256_hex(b"")


def test_frame_hash_column_order_invariant():
    a = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    b = a[["y", "x"]]
    assert frame_sha256(a) == frame_sha256(b)


def test_frame_hash_changes_with_values_and_row_order():
    a = pd.DataFrame({"x": [1.0, 2.0]})
    assert frame_sha256(a) != frame_sha256(pd.DataFrame({"x": [1.0, 2.5]}))
    assert frame_sha256(a) != frame_sha256(pd.DataFrame({"x": [2.0, 1.0]}))


def test_frame_hash_handles_missing_values():
    a = pd.DataFrame({"x": [1.0, None], "y": ["a", None]})
    assert frame_sha256(a) == frame_sha256(a.copy())


def test_frame_hash_int_and_float_equal_values_match():
    # CSV round trips can turn 1 into 1.0; the hash should describe values, not dtypes.
    assert frame_sha256(pd.DataFrame({"x": [1, 2]})) == frame_sha256(pd.DataFrame({"x": [1.0, 2.0]}))
