from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    src_path = str(src_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
