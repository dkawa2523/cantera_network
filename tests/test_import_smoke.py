import importlib


def test_import_smoke() -> None:
    importlib.import_module("rxn_platform")
