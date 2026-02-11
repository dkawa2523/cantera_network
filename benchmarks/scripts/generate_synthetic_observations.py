#!/usr/bin/env python3
"""Generate synthetic observation CSV for assimilation benchmark.

This script is intentionally lightweight and does NOT depend on the platform's internal artifacts.
It generates observation points based on the condition CSV and user-chosen QoIs.

If your platform already has a dedicated observation generator, you can ignore this script.

Output schema (CSV):
  case_id, observable, value, sigma

Observables used here (default):
  - CO2_final
  - CO_final
  - T_final

NOTE: This script can optionally call your platform to generate baseline simulations, but by default
it only prepares a template with placeholder values. Replace placeholders by running the
`bench_gri30_sim_sweep` pipeline and exporting values, or extend this script.
"""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path

DEFAULT_OBS = ["CO2_final", "CO_final", "T_final"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="gri30_small | gri30_medium | gri30_wide")
    ap.add_argument("--conditions", default=None, help="Override conditions CSV path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", type=float, default=0.03, help="relative noise for mole fractions; absolute for temperature if >1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--observables", nargs="+", default=DEFAULT_OBS)
    args = ap.parse_args()

    # Resolve default condition files
    if args.conditions is None:
        args.conditions = f"benchmarks/assets/conditions/{args.case}.csv" if args.case.startswith("gri30_") else f"benchmarks/assets/conditions/{args.case}.csv"
    cond_path = Path(args.conditions)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(args.seed)
    rows = []
    with cond_path.open("r", encoding="utf-8") as f:
        cr = csv.DictReader(f)
        for r in cr:
            cid = r["case_id"]
            for obs in args.observables:
                # Placeholder value: you should replace using baseline simulation results
                val = 1.0 if obs != "T_final" else 1500.0
                sigma = args.noise * val if obs != "T_final" else max(5.0, args.noise)
                # Add noise to create synthetic "measured" value
                noisy = val + rnd.gauss(0.0, sigma)
                rows.append({"case_id": cid, "observable": obs, "value": noisy, "sigma": sigma})

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id","observable","value","sigma"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] wrote synthetic observation template: {out_path}")
    print("NOTE: Values are placeholders unless you plug in baseline simulation outputs.")

if __name__ == "__main__":
    main()
