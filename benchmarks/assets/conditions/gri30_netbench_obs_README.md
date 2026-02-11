# gri30_netbench_obs.csv

`gri30_netbench_obs.csv` は、同化ベンチ用の観測（合成）データです。

生成手順（推奨：Cantera直で生成）:

```bash
python benchmarks/scripts/netbench_generate_obs_cantera.py \
  --mechanism benchmarks/assets/mechanisms/gri30.yaml \
  --conditions benchmarks/assets/conditions/gri30_netbench_train.csv \
  --truth benchmarks/assets/truth/gri30_truth_multipliers.json \
  --out benchmarks/assets/observations/gri30_netbench_obs.csv
```

出力スキーマ:
- case_id
- observable  (CO2_final / CO_final / T_peak / ignition_delay)
- value
- sigma
