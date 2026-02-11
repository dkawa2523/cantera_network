# 03_TEMPORAL_NETWORK_SPEC（時系列フラックスグラフ仕様）

TemporalFluxGraph は `graphs/species_graph/layer_*.npz` と `graphs/meta.json` を持つ。
`meta.json` は `graph.json` と同内容で、最低限以下を満たす。

## 必須フィールド
- `kind`: `"temporal_flux"`
- `source.run_ids`: 参照した run_id の配列
- `species.count` / `species.order`
- `reactions.count` / `reactions.order`
- `windowing.type` / `windowing.count`
- `species_graph.path`: `"species_graph"`
- `species_graph.layers`: 各windowのメタ

## layer メタ（species_graph.layers）
- `index`: window index
- `path`: `species_graph/layer_XXX.npz`
- `nnz`: 非ゼロ要素数
- `density`: 密度
- `window.start_idx` / `window.end_idx`
- `window.start_time` / `window.end_time`

## 備考
- `graphs/meta.json` は `graphs/graph.json` の互換コピーとして扱う。
- 旧形式の読み込みは互換として許容し、新規書き込みは本仕様のみ。
- 現行実装は multi-run（`source.run_ids` が複数）の場合、各 run の `coords.time` が **完全一致**していることを要求する（長さ・各時刻値の一致）。不一致の場合はエラーになる。
