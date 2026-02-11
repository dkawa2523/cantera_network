# Task: Contract/Smoke/Regressionテストの再整備

## Summary
- **Goal**: 新構成に合わせてsmoke/contract/regressionテストを再整備する。
- **Scope**: RunStore出力、viz出力、graph出力、reduction出力のテスト。
- **Non-goals**: 重い学習をフルに回すCI。

## Depends on
- depends_on: T086R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/06_TESTING_AND_QA.md

## Acceptance Criteria
- [ ] `pytest -q` が通る（軽量構成）。
- [ ] RunStore必須ファイルが存在することをテストで保証。
- [ ] vizの主要出力が存在することをテストで保証。

## Verification
Commands to run (examples):
- `pytest -q`

## Notes
- 大きいデータはミニデータで代替。
