# Task: TemporalFluxGraph共通化（graphs/temporal_flux_graph.py）

## Summary
- **Goal**: TemporalFluxGraph生成を単一モジュールへ集約し、全手法共通部品にする。
- **Scope**: graph生成APIの統一、`graphs/meta.json`の標準化、キャッシュキーの固定。
- **Non-goals**: 二部グラフの完全刷新（最小限の互換維持）。

## Depends on
- depends_on: T080R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/02_ARTIFACT_CONTRACTS.md
- docs/03_TEMPORAL_NETWORK_SPEC.md

## Acceptance Criteria
- [ ] graph生成の入口が単一化される。
- [ ] `graphs/species_graph/layer_*.npz` + `graphs/meta.json` が生成される。
- [ ] CNR/AMORE/GNNから同一APIで利用できる。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli run recipe=build_temporal_graph run_id=demo01`

## Notes
- 既存出力形式は互換読み込みのみ残す。
