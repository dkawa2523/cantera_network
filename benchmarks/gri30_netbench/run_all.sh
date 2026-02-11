#!/usr/bin/env bash
set -euo pipefail

MECH="benchmarks/assets/mechanisms/gri30.yaml"
EXP="netbench"
CASE="${CASE:-tr000}"

echo "[1/6] Setup mechanisms (if not yet)"
python3 benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms || true

echo "[2/6] Graph build (train)"
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_graph \
  run_id=nb_graph_${CASE} exp=${EXP} \
  sim=cantera_0d mechanism.path=${MECH} +benchmarks=gri30_netbench_train benchmarks.case_id=${CASE}

echo "[3/6] Reduction (ROP threshold)"
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_rop \
  run_id=nb_reduce_rop_${CASE} exp=${EXP} \
  sim=cantera_0d mechanism.path=${MECH} +benchmarks=gri30_netbench_train benchmarks.case_id=${CASE}

echo "[4/6] Reduction (centrality)"
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_centrality \
  run_id=nb_reduce_centrality_${CASE} exp=${EXP} \
  sim=cantera_0d mechanism.path=${MECH} +benchmarks=gri30_netbench_train benchmarks.case_id=${CASE}

echo "[5/6] Generate synthetic obs"
python3 benchmarks/scripts/netbench_generate_obs_cantera.py \
  --mechanism ${MECH} \
  --conditions benchmarks/assets/conditions/gri30_netbench_train.csv \
  --truth benchmarks/assets/truth/gri30_truth_multipliers.json \
  --out benchmarks/assets/observations/gri30_netbench_obs.csv

echo "[6/6] Assimilation EKI"
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_assim_eki \
  run_id=nb_assim_eki_${CASE} exp=${EXP} \
  sim=cantera_0d mechanism.path=${MECH} +benchmarks=gri30_netbench_train benchmarks.case_id=${CASE} \
  assimilation.obs_file=benchmarks/assets/observations/gri30_netbench_obs.csv

echo "[DONE]"
