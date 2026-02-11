import json
import logging
from pathlib import Path

import pytest

from rxn_platform import doctor


def _write_run_failure(run_root: Path, *, error: str) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_root.name,
        "exp": run_root.parent.name,
        "created_at": "2026-02-01T00:00:00Z",
        "recipe": "smoke",
        "store_root": str(run_root / "artifacts"),
        "simulator": "dummy",
        "mechanism_hash": "deadbeef",
        "conditions_hash": "deadbeef",
        "qoi_spec_hash": "deadbeef",
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_root / "config_resolved.yaml").write_text("{}\n", encoding="utf-8")
    metrics = {
        "schema_version": 1,
        "status": "failed",
        "error": error,
    }
    (run_root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_doctor_logs_failure_category(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    runstore_root = tmp_path / "runs"
    run_root = runstore_root / "exp" / "fail1"
    _write_run_failure(run_root, error="BackendError: cantera failed")

    doctor_logger = logging.getLogger("rxn_platform.doctor.test")
    doctor_logger.setLevel(logging.INFO)
    doctor_logger.propagate = True
    doctor_logger.handlers.clear()

    with caplog.at_level(logging.INFO, logger=doctor_logger.name):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(doctor, "RUNS_ROOT", runstore_root)
            doctor.run_doctor(
                config_path=Path(__file__).resolve().parents[1] / "configs",
                config_name="default",
                overrides=[f"store.root={tmp_path / 'artifacts'}"],
                strict=False,
                logger=doctor_logger,
            )

    assert any(
        "category=sim" in record.getMessage() for record in caplog.records
    ), "expected doctor to log a sim category"
