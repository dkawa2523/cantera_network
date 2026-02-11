import pytest

from rxn_platform.registry import Registry


def test_register_get_list_roundtrip() -> None:
    registry = Registry()
    sentinel = object()

    registry.register("backend", "dummy", sentinel)

    assert registry.get("backend", "dummy") is sentinel
    assert registry.list("backend") == ["dummy"]


def test_unknown_kind_error_is_clear() -> None:
    registry = Registry()

    with pytest.raises(KeyError) as exc:
        registry.get("unknown", "dummy")

    message = str(exc.value)
    assert "Unknown registry kind" in message
    assert "unknown" in message


def test_unknown_name_error_is_clear() -> None:
    registry = Registry()
    registry.register("task", "t1", object())

    with pytest.raises(KeyError) as exc:
        registry.get("task", "missing")

    message = str(exc.value)
    assert "not registered" in message
    assert "task" in message


def test_duplicate_registration_requires_overwrite() -> None:
    registry = Registry()
    registry.register("feature", "f1", 1)

    with pytest.raises(ValueError) as exc:
        registry.register("feature", "f1", 2)

    assert "already registered" in str(exc.value)

    registry.register("feature", "f1", 2, overwrite=True)
    assert registry.get("feature", "f1") == 2
