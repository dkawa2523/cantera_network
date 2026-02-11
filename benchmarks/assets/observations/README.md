# observations/

同化ベンチ（B6）用の観測データを置きます。

## 推奨フロー
1. まず `bench_gri30_sim_sweep` を回して baseline の run を作る
2. baseline の出力から（CO2_final, CO_final, T_final など）を抽出する
3. ノイズを付与して `gri30_obs.csv` を作る

簡易にはテンプレ生成だけ可能です：

```bash
python benchmarks/scripts/generate_synthetic_observations.py \
  --case gri30_small \
  --out benchmarks/assets/observations/gri30_obs.csv
```

※このテンプレは placeholder 値を含むため、正しい評価には baseline 値の差し替えが必要です。
