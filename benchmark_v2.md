# GRI30 Multicase 実行結果まとめ（v2）

## 実行コマンド

今回実行したコマンドは以下です（GNN を含む multicase 評価）。

```bash
/Volumes/SP\ PX10/Main_code/cantera/cantera/.venv/bin/python -m rxn_platform.cli run \
  pipeline=bench_gri30_multicase_rop_gnn_cover_restore \
  exp=netbench run_id=nb_multicase_cover_restore_small_v3 \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_small
```

## この実行で何をしたか

- `gri30_small` の複数条件を `sim.sweep_csv` でまとめて実行
- その複数条件から ROP/GNN の重要度を作成
- 強めの縮退（`top_k=60`）を実施
- `reduction.validate(case_mode=all)` で複数条件 QoI を評価
- 失敗ケースをカバーするように `reduction.repair_cover_restore` で反応を復帰
- 復帰後パッチを再度複数条件で検証

## 主な結果（言葉で要約）

- 修復前は、ROP/GNN ともに `pass_rate=0.895833`（48項目中 43 合格）で、5項目が不合格でした。
- `repair_cover_restore` が 12 反応を復帰し、無効化反応数は `262 -> 250` になりました。
- 修復後は、ROP/GNN ともに `pass_rate=0.9375`（48項目中 45 合格）へ改善しました。
- 最大相対誤差も `0.967403 -> 0.963968`、平均相対誤差も改善しており、精度劣化を抑えつつ復帰できています。

## 数値比較（修復前後）

- ROP base: `pass_rate=0.895833`, `max_rel=0.967403`, `mean_rel=0.136034`
- ROP repaired: `pass_rate=0.937500`, `max_rel=0.963968`, `mean_rel=0.119485`
- GNN base: `pass_rate=0.895833`, `max_rel=0.967403`, `mean_rel=0.134663`
- GNN repaired: `pass_rate=0.937500`, `max_rel=0.963968`, `mean_rel=0.118662`

## 復帰された反応

- 復帰数: `12`
- 反応 index:
  - `[30, 99, 131, 134, 143, 144, 152, 165, 264, 279, 283, 289]`

## 成果物ディレクトリ

- Run root:
  - `runs/netbench/nb_multicase_cover_restore_small_v3`
- validation 結果:
  - `runs/netbench/nb_multicase_cover_restore_small_v3/artifacts/validation/8c772e92d8fe06a6`（ROP base）
  - `runs/netbench/nb_multicase_cover_restore_small_v3/artifacts/validation/c8c598e53a6381a4`（ROP repaired）
  - `runs/netbench/nb_multicase_cover_restore_small_v3/artifacts/validation/82d032d3ece02304`（GNN base）
  - `runs/netbench/nb_multicase_cover_restore_small_v3/artifacts/validation/45c9d7f841a74a96`（GNN repaired）
- repair patch:
  - `runs/netbench/nb_multicase_cover_restore_small_v3/artifacts/reduction/b56b7592c981ac1d`
  - `runs/netbench/nb_multicase_cover_restore_small_v3/artifacts/reduction/30dbe7c52dc4d90f`

## 補足

- 実行中に出る Zarr の `UnstableSpecificationWarning` は、今回の計算成功/失敗そのものとは別の警告です（計算は完了し、成果物も生成済み）。
