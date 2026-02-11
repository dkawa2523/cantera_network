#!/usr/bin/env python3
"""Compatibility wrapper for tools/codex_loop/run_loop.py."""

from pathlib import Path
import sys


repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from tools.codex_loop.run_loop import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
