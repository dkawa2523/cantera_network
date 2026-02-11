"""Local import shim for the src/ package layout."""

from pathlib import Path

_src_pkg = Path(__file__).resolve().parent.parent / "src" / "rxn_platform"
if _src_pkg.is_dir():
    _src_str = str(_src_pkg)
    if _src_str not in __path__:
        __path__.append(_src_str)
    del _src_str

del _src_pkg, Path

__version__ = "0.0.0"

__all__ = ["__version__"]
