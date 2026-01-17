# T041 Optimization v0: random search（条件 or multipliers）+ history artifact

- **id**: `T041`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T024, T014, T025, T020
- **unblocks**: (none)
- **skills**: python, optimization

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

最適化・同化・縮退評価は複数回のCantera実行（またはdummy実行）を前提とする。
まずはベンチマークとして random search を実装し、history artifact と可視化の基準を作る。

## Scope

- 探索パラメータ（例: multipliers, 初期T/P, flow等）を config で定義
- randomにサンプルし pipeline を実行して目的関数（Observable）を評価
- 履歴（params, objective, run_id）を OptimizationArtifact として保存
- 再現性のため seed を固定し、同設定で同じ履歴になるようにする

## Out of scope

- 高性能なBO/GA（後回し）

## Deliverables

- `src/rxn_platform/tasks/optimization.py`（RandomSearchOptimizer）
- `tests/test_opt_random_dummy.py`

## Acceptance Criteria

- dummyで短時間に最適化履歴が生成される
- 履歴artifactが保存され、再実行で再現する

## Verification

```bash
$ pytest -q
```

## Notes

- (none)

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

