from pathlib import Path

import pytest

from rxn_platform.core import make_artifact_id, make_run_id, stable_hash


def test_stable_hash_dict_order() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_stable_hash_list_tuple_equivalence() -> None:
    assert stable_hash({"values": [1, 2]}) == stable_hash({"values": (1, 2)})


def test_stable_hash_path_and_numbers() -> None:
    cfg = {"path": Path("configs/default.yaml"), "seed": 123}
    assert stable_hash(cfg) == stable_hash({"path": "configs/default.yaml", "seed": 123})
    assert stable_hash({"value": 1}) != stable_hash({"value": 2})


def test_stable_hash_exclude_keys() -> None:
    base = {"a": 1, "b": 2}
    with_log = {"a": 1, "b": 2, "log": {"level": "INFO"}}
    assert stable_hash(with_log, exclude_keys={"log"}) == stable_hash(base)
    assert stable_hash(with_log) != stable_hash(base)


def test_stable_hash_numpy_scalars() -> None:
    np = pytest.importorskip("numpy")
    cfg = {"a": np.int64(1), "b": np.float32(2.5)}
    assert stable_hash(cfg) == stable_hash({"a": 1, "b": 2.5})


def test_make_ids_change_with_content() -> None:
    assert make_run_id({"a": 1}) != make_run_id({"a": 2})
    assert make_artifact_id(inputs={"x": 1}, config={}) != make_artifact_id(
        inputs={"x": 2},
        config={},
    )
