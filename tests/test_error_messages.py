import logging

import pytest

from rxn_platform.core import load_config
from rxn_platform.errors import ArtifactError, BackendError, ConfigError
from rxn_platform.logging_utils import log_exception, run_with_error_handling
from rxn_platform.registry import Registry, resolve_backend
from rxn_platform.store import ArtifactStore


def test_missing_config_raises_config_error(tmp_path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError) as exc:
        load_config(missing)

    message = str(exc.value)
    assert "Config not found" in message
    assert str(missing) in message


def test_missing_artifact_raises_artifact_error(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactError) as exc:
        store.open_manifest("runs", "missing")

    message = str(exc.value)
    assert "Manifest not found" in message
    assert "manifest.yaml" in message


def test_missing_backend_raises_backend_error() -> None:
    registry = Registry()

    with pytest.raises(BackendError) as exc:
        resolve_backend("missing-backend", registry=registry)

    message = str(exc.value)
    assert "Backend" in message
    assert "missing-backend" in message


def test_run_with_error_handling_logs_and_reraises(caplog) -> None:
    logger = logging.getLogger("rxn_platform.test")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    logger.handlers.clear()

    def _raise_config_error() -> None:
        raise ConfigError("Config not found: configs/missing.yaml")

    with caplog.at_level(logging.INFO, logger=logger.name):
        with pytest.raises(ConfigError):
            run_with_error_handling(_raise_config_error, logger=logger)

    assert any(
        "Config not found" in record.getMessage() for record in caplog.records
    )


def test_log_exception_emits_traceback_at_debug_level(caplog) -> None:
    logger = logging.getLogger("rxn_platform.test.debug")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    logger.handlers.clear()

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        log_exception(logger, ConfigError("Config not found: configs/missing.yaml"))

    assert any(record.exc_info for record in caplog.records)
