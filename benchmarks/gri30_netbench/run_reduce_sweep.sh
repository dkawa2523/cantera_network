#!/usr/bin/env bash
set -euo pipefail

MECH="${MECH:-benchmarks/assets/mechanisms/gri30.yaml}"
EXP="${EXP:-netbench}"
BENCH="${BENCH:-gri30_netbench_train}"
CASE="${CASE:-tr000}"
VAL_CONDITIONS="${VAL_CONDITIONS:-}"

echo "[1/3] Setup mechanisms (if not yet)"
python3 benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms || true

OVERRIDES=(
  "sim=cantera_0d"
  "mechanism.path=${MECH}"
  "+benchmarks=${BENCH}"
  "benchmarks.case_id=${CASE}"
)
if [[ -n "${VAL_CONDITIONS}" ]]; then
  OVERRIDES+=("benchmarks.validation_conditions_file=${VAL_CONDITIONS}")
fi

echo "[2/3] Reduction sweep (ROP threshold)"
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_rop_sweep \
  "run_id=nb_reduce_rop_sweep_${CASE}" "exp=${EXP}" \
  "${OVERRIDES[@]}"

echo "[3/3] Reduction sweep (centrality)"
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_centrality_sweep \
  "run_id=nb_reduce_centrality_sweep_${CASE}" "exp=${EXP}" \
  "${OVERRIDES[@]}"

echo "[DONE]"

