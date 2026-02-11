# Task: CLIのrun体験統一（recipe + run_id）

## Summary
- **Goal**: すべての実行入口を `rxn cli run recipe=... run_id=...` に統一する。
- **Scope**: recipeベースの入口整理、不要なサブコマンドの整理、ヘルプ表示の明確化。
- **Non-goals**: 既存CLIの完全削除。互換的に残すが推奨経路はrunに統一。

## Depends on
- depends_on: T081R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/03_CONFIG_CONVENTIONS.md
- docs/04_CONFIG_SIMPLIFICATION.md

## Acceptance Criteria
- [ ] `rxn cli run recipe=... run_id=...` が主要ユースケースを網羅する。
- [ ] 旧コマンドは警告付きで残し、docsはrunベースに統一される。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli help`
- `python -m rxn_platform.cli run recipe=sim_sweep run_id=demo01`

## Notes
- 旧CLIは段階的にdeprecated扱いとし、P2で削除検討。
