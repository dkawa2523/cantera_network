# Task: RunStore出力レイアウトの一本化（timeseries/graphs/vizの標準化）

## Summary
- **Goal**: すべての出力を `runs/<exp>/<run_id>/` に統一し、必須ファイル＋標準サブディレクトリに揃える。
- **Scope**: sim出力の `sim/timeseries.zarr` 統一、graph出力の `graphs/species_graph/layer_*.npz` + `graphs/meta.json`、vizの `viz/` 統一。
- **Non-goals**: 既存のartifact store構造の完全削除。互換読み込みは別タスクで扱う。

## Depends on
- depends_on: T061G

## Contracts / References
- docs/00_INVARIANTS.md
- docs/02_ARTIFACT_CONTRACTS.md
- docs/03_CONFIG_CONVENTIONS.md
- docs/04_CONFIG_SIMPLIFICATION.md

## Acceptance Criteria
- [ ] RunStore直下に `manifest.json`, `config_resolved.yaml`, `metrics.json` が常に生成される。
- [ ] 時系列出力が `sim/timeseries.zarr` に統一される（既存 `state.zarr` は互換読み込みのみ）。
- [ ] グラフ出力が `graphs/species_graph/layer_*.npz` と `graphs/meta.json` を持つ。
- [ ] 可視化出力が `viz/` 配下に統一される。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli run recipe=sim_sweep run_id=demo01`
- `python -m rxn_platform.cli run recipe=build_temporal_graph run_id=demo01`
- `python -m rxn_platform.cli run recipe=viz_report run_id=demo01`

## Notes
- 互換読み込みは警告付きで残し、書き込みは新規レイアウトのみ。
