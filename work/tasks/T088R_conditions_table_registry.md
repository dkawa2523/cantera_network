# Task: 条件テーブル（cases）管理の整理

## Summary
- **Goal**: caseテーブルをRunStoreの唯一の入力キーとして明確化し、参照/再利用を簡素化する。
- **Scope**: conditionsの保存形式統一、case_id参照の一本化、dataset registry連携。
- **Non-goals**: DVC等の外部ストレージ統合。

## Depends on
- depends_on: T080R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/02_ARTIFACT_CONTRACTS.md

## Acceptance Criteria
- [ ] caseテーブルがRunStoreから参照できる。
- [ ] case_idが唯一の入力キーとして統一される。
- [ ] dataset registryにcaseテーブルのhashが記録される。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli run recipe=sim_sweep run_id=demo01`
- `python -m rxn_platform.cli dataset register --run-id demo01 --exp default`

## Notes
- caseテーブル形式はparquet/csvのどちらかに統一する。
