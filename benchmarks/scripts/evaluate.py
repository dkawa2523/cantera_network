#!/usr/bin/env python3
"""Evaluate benchmark outputs by scanning artifacts directory.

This evaluator is conservative: it checks for existence of expected artifact kinds and writes a summary report.
If your artifact layout differs, edit `ARTIFACT_PATTERNS`.

Usage:
  python benchmarks/scripts/evaluate.py --artifacts artifacts --out benchmarks/reports/summary.md
"""
from __future__ import annotations
import argparse
from pathlib import Path
import datetime

ARTIFACT_PATTERNS = {
  "runs": ["runs/*/manifest.yaml"],
  "observables": ["observables/*/manifest.yaml", "observables/*/observables.parquet"],
  "graphs": ["graphs/*/manifest.yaml", "graphs/*/graph.json"],
  "features": ["features/*/manifest.yaml", "features/*/features.parquet"],
  "sensitivity": ["sensitivity/*/manifest.yaml", "sensitivity/*/sensitivity.parquet"],
  "models": ["models/*/manifest.yaml"],
  "reports": ["reports/*/manifest.yaml", "reports/*/index.html"],
}

def glob_any(root: Path, patterns: list[str]) -> list[Path]:
    out = []
    for pat in patterns:
        out.extend(root.glob(pat))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="artifact root (default: artifacts)")
    ap.add_argument("--out", required=True, help="output markdown report path")
    args = ap.parse_args()

    root = Path(args.artifacts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Benchmark Summary\n")
    lines.append(f"- Generated at: {datetime.datetime.utcnow().isoformat()}Z\n")
    lines.append(f"- Artifact root: `{root}`\n\n")

    ok = True
    for kind, pats in ARTIFACT_PATTERNS.items():
        hits = glob_any(root, pats)
        lines.append(f"## {kind}\n")
        if hits:
            lines.append(f"- Found: **{len(hits)}** files (patterns: {pats})\n")
        else:
            ok = False
            lines.append(f"- Found: **0** files (patterns: {pats})\n")
            lines.append(f"- NOTE: This may be OK if you didn't run that category yet, or your layout differs.\n")
        lines.append("\n")

    lines.append("## Next Actions\n")
    if ok:
        lines.append("- All expected artifact types have at least one file. Consider adding metric-level evaluation (RMSE, speedup) next.\n")
    else:
        lines.append("- Some artifact types were not detected. Run the relevant pipelines from `benchmarks/BENCHMARKS.md`.\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] wrote report: {out_path}")

if __name__ == "__main__":
    main()
