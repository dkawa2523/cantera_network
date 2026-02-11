# Task: 可視化刷新（ネットワーク系）

## Summary
- **Goal**: 反応ネットワーク解析に有用な図のみを標準化し、読める出力にする。
- **Scope**: species graph / bipartite / quotient / diff の4種を実装、上位のみ描画。
- **Non-goals**: 全描画・過剰な図の生成。出力過多は禁止。

## Depends on
- depends_on: T084R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/05_VISUALIZATION_STANDARDS.md

## Acceptance Criteria
- [ ] `viz/network/` に4種の図が出力される。
- [ ] 上位ノード/エッジのみ描画する制限がある。
- [ ] ファイル命名規約が統一される。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli run recipe=viz_report run_id=demo01`

## Notes
- 出力はPNG/SVG/HTMLのいずれかで統一。
