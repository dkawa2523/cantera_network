"""Metadata helpers shared across tasks and pipelines."""

from __future__ import annotations

from pathlib import Path
import platform
import subprocess
from typing import Any

from rxn_platform import __version__


def code_metadata() -> dict[str, Any]:
    payload: dict[str, Any] = {"version": __version__}
    git_dir = Path.cwd() / ".git"
    if not git_dir.exists():
        return payload
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        payload["git_commit"] = commit
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        payload["dirty"] = bool(dirty)
    except (OSError, subprocess.SubprocessError):
        return payload
    return payload


def provenance_metadata() -> dict[str, Any]:
    return {"python": platform.python_version()}


__all__ = ["code_metadata", "provenance_metadata"]
