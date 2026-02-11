# Task: Structured Config化とyaml最小化（default + recipes）

## Summary
- **Goal**: 設定の真実を dataclass（structured config）に置き、yamlを default + recipes に縮退する。
- **Scope**: config構造の整理、recipeの短縮、legacy groupの段階的互換。
- **Non-goals**: すべての旧yamlの即時削除。移行期間は互換読み込みを維持。

## Depends on
- depends_on: T080R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/03_CONFIG_CONVENTIONS.md
- docs/04_CONFIG_SIMPLIFICATION.md

## Acceptance Criteria
- [ ] default + recipes の2層構成で主要レシピが動く。
- [ ] structured configがsource of truthとなる（yamlは上書きのみ）。
- [ ] legacy groupは互換読み込みのみで、新規追加は禁止。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli cfg`
- `python -m rxn_platform.cli run recipe=sim_sweep run_id=demo01`

## Notes
- structured config定義は `rxn_platform/config/` に集約する。
