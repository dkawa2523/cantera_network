# GRI30 NetBench（325反応）パック

- ネットワーク構築（Graph）
- 縮退（Reduction）を train/val で検証
- データ同化（Assimilation）を合成観測で評価

を同一テストケースで回せるようにした追加ファイルです。

## 追加されるファイル
- `benchmarks/assets/conditions/gri30_netbench_{train,val}.csv`
- `configs/benchmarks/gri30_netbench_{train,val}.yaml`
- `configs/pipeline/bench_gri30_netbench_*.yaml`
- `benchmarks/scripts/netbench_generate_obs_cantera.py`
- `benchmarks/assets/truth/gri30_truth_multipliers.json`

## 使い方（最短）
1) mechanism を用意（gri30.yaml）
2) graph build
3) reduction を2方式で実行
4) 合成観測生成
5) 同化（EKI/ES-MDA）実行

詳細は `NETBENCH.md` を参照してください。
