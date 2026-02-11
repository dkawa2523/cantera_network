#!/usr/bin/env python3
"""Copy Cantera-supplied mechanism files into benchmarks/assets/mechanisms.

This script tries, in order:
1) Import cantera and use `cantera.get_data_directories()` to locate `gri30.yaml`
2) If not found, optionally download from Cantera GitHub (requires network).

Usage:
  python benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms
  python benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms --download
"""
from __future__ import annotations
import argparse, shutil, sys, os
from pathlib import Path
import urllib.request

CAN_GITHUB_RAW = {
  "gri30.yaml": "https://raw.githubusercontent.com/Cantera/cantera/main/data/gri30.yaml",
  "gri30_ion.yaml": "https://raw.githubusercontent.com/Cantera/cantera/main/data/gri30_ion.yaml",
  "gri30_highT.yaml": "https://raw.githubusercontent.com/Cantera/cantera/main/data/gri30_highT.yaml",
}

def _data_dirs_from_env() -> list[Path]:
    raw = os.environ.get("CANTERA_DATA")
    if not raw:
        return []
    return [Path(entry) for entry in raw.split(os.pathsep) if entry]


def _data_dirs_from_cantera() -> list[Path]:
    try:
        import cantera as ct
    except Exception:
        return []
    return [Path(entry) for entry in ct.get_data_directories()]


def copy_from_cantera(dest: Path) -> list[str]:
    paths = []
    data_dirs: list[Path] = []
    seen: set[str] = set()
    for source in (_data_dirs_from_cantera(), _data_dirs_from_env()):
        for entry in source:
            entry_str = str(entry)
            if entry_str in seen:
                continue
            seen.add(entry_str)
            data_dirs.append(entry)
    for dp in data_dirs:
        if dp.exists():
            for name in CAN_GITHUB_RAW.keys():
                p = dp / name
                if p.exists():
                    shutil.copy2(p, dest / name)
                    paths.append(str(dest / name))
    return paths

def download_from_github(dest: Path) -> list[str]:
    out = []
    for name, url in CAN_GITHUB_RAW.items():
        target = dest / name
        try:
            with urllib.request.urlopen(url) as r:
                target.write_bytes(r.read())
            out.append(str(target))
        except Exception as e:
            print(f"[WARN] failed download {name}: {e}", file=sys.stderr)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=str, required=True)
    ap.add_argument("--download", action="store_true", help="Download from Cantera GitHub if not available locally")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    existing = {name for name in CAN_GITHUB_RAW.keys() if (dest / name).exists()}
    if existing == set(CAN_GITHUB_RAW.keys()):
        print("[OK] mechanisms already present:\n- " + "\n- ".join(str(dest / n) for n in sorted(existing)))
        return
    if existing:
        print("[INFO] mechanisms already present (partial):\n- " + "\n- ".join(str(dest / n) for n in sorted(existing)))

    copied = copy_from_cantera(dest)
    present = existing | {Path(p).name for p in copied}
    if present == set(CAN_GITHUB_RAW.keys()):
        print("[OK] mechanisms ready:\n- " + "\n- ".join(str(dest / n) for n in sorted(present)))
        return
    if copied:
        print("[OK] copied from cantera data directories:\n- " + "\n- ".join(copied))
        return

    print("[INFO] cantera not importable or mechanisms not found in data directories.")
    if args.download:
        downloaded = download_from_github(dest)
        present = present | {Path(p).name for p in downloaded}
        if present == set(CAN_GITHUB_RAW.keys()):
            print("[OK] downloaded from Cantera GitHub:\n- " + "\n- ".join(downloaded))
            return
        if downloaded:
            print("[WARN] downloaded some mechanisms but others are missing.", file=sys.stderr)
        missing = sorted(set(CAN_GITHUB_RAW.keys()) - present)
        if missing:
            print("[ERROR] missing mechanisms: " + ", ".join(missing), file=sys.stderr)
        else:
            print("[ERROR] could not download any mechanisms. Check network or URLs.", file=sys.stderr)
            sys.exit(2)
    else:
        missing = sorted(set(CAN_GITHUB_RAW.keys()) - present)
        if missing:
            print("[HINT] Missing mechanisms: " + ", ".join(missing))
        print("[HINT] Install Cantera or set CANTERA_DATA, then re-run. Or use --download.")
        sys.exit(1)

if __name__ == "__main__":
    main()
