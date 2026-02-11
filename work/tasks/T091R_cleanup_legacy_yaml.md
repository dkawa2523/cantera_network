# Task: Legacy yamlの整理と移行ガイド

## Summary
- **Goal**: legacy yamlの整理を行い、不要なgroup乱立を止める。
- **Scope**: 非推奨yamlの一覧化、移行ガイド、削除対象の明確化。
- **Non-goals**: 即時削除。移行計画に沿って段階対応。

## Depends on
- depends_on: T081R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/03_CONFIG_CONVENTIONS.md
- docs/04_CONFIG_SIMPLIFICATION.md

## Acceptance Criteria
- [ ] legacy yamlの一覧と移行手順がdocsに記載される。
- [ ] 新規yaml追加は禁止される（READMEに明記）。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli cfg`

## Notes
- 互換読み込みの警告を追加する。
