#!/usr/bin/env bash
set -euo pipefail

echo "This is a convenience helper. Edit commands if your CLI differs."
echo ""

echo "1) Setup mechanisms"
echo "python3 benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms"
echo ""

echo "2) Dummy smoke"
echo "python3 -m rxn_platform.cli run pipeline=bench_dummy_smoke run_id=bench_dummy_smoke exp=bench sim=dummy"
echo ""

echo "3) GRI30 network bench"
echo "python3 -m rxn_platform.cli run pipeline=bench_gri30_network run_id=bench_gri30_network exp=bench sim=cantera_0d sim.mechanism=benchmarks/assets/mechanisms/gri30.yaml"
echo ""

echo "4) Summarize artifacts"
echo "python3 benchmarks/scripts/evaluate.py --artifacts artifacts --out benchmarks/reports/summary.md"
