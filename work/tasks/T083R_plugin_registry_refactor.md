# Task: Methods/Viz/QoIのプラグイン化（if/else増殖抑止）

## Summary
- **Goal**: 縮退/学習/可視化/QoIを registry 経由で解決し、if/else分岐を廃止する。
- **Scope**: registryインターフェース整理、プラグイン登録手順の統一、既存実装の移設。
- **Non-goals**: 実装の全面刷新。入口と登録方式の統一が目的。

## Depends on
- depends_on: T081R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/02_ARTIFACT_CONTRACTS.md

## Acceptance Criteria
- [ ] methods/viz/qoiが registry 経由で解決される。
- [ ] if/elseによる手法切替を削除または非推奨化。
- [ ] 追加手順が docs に明記される。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli run recipe=reduce_cnr_coarse run_id=demo01`
- `python -m rxn_platform.cli run recipe=viz_report run_id=demo01`

## Notes
- registryは既存 `rxn_platform.registry` を拡張する。
