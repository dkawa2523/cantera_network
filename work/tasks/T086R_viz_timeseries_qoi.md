# Task: 可視化刷新（時系列/QoI/縮退比較）

## Summary
- **Goal**: 時系列・QoIの可視化を標準セットとして整備する。
- **Scope**: top-k種組成、主要フラックス、QoI時系列、縮退前後比較の散布図。
- **Non-goals**: すべての変数の網羅。標準セットのみ。

## Depends on
- depends_on: T080R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/05_VISUALIZATION_STANDARDS.md

## Acceptance Criteria
- [ ] `viz/timeseries/` と `viz/reduction/` に標準図が出力される。
- [ ] top-k選択が可能で、図が読めるサイズに制限される。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli run recipe=viz_report run_id=demo01`

## Notes
- plotの命名規約を明記する。
