# Task: Reduction手法のプラグイン統合（CNR/AMORE/LearnCK/GNN）

## Summary
- **Goal**: 主要縮退手法を共通インターフェースで統合し、入口を保持したまま整理する。
- **Scope**: registry登録の整理、共通I/O、設定の統一。
- **Non-goals**: 手法のアルゴリズム刷新。

## Depends on
- depends_on: T083R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/02_ARTIFACT_CONTRACTS.md
- docs/02_REDUCTION_METHODS.md

## Acceptance Criteria
- [ ] CNR/AMORE/LearnCK/GNNの入口が同一インターフェースで登録される。
- [ ] if/elseでの切替が削除される。
- [ ] RunStore出力が統一される。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli run recipe=reduce_cnr_coarse run_id=demo01`
- `python -m rxn_platform.cli run recipe=reduce_amore_search run_id=demo01`
- `python -m rxn_platform.cli run recipe=reduce_learnck_style run_id=demo01`

## Notes
- optional依存（GNN）はextrasで維持。
