# T054 Assimilation: Laplacian regularizer（ペナルティ/疑似観測）をEKIに統合（オプション）

- **id**: `T054`
- **priority**: `P3`
- **status**: `todo`
- **depends_on**: T053, T044
- **unblocks**: (none)
- **skills**: python, data-assimilation

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

グラフラプラシアン正則化は、反応ネットワークの“連続性/妥当性”を同化に注入する有力な方法。
EKIの更新に Laplacian penalty（疑似観測）を組み込み、過度に独立に振れない推定を可能にする。

## Scope

- T053で計算したLを用い、パラメータベクトルに対するペナルティ ||L p||^2 を定義
- EKIの観測ベクトルに pseudo-observation を追加する方法（または updateでの正則化）を実装
- 正則化強度 lambda を config で指定できる
- lambda=0 で通常EKIに戻る

## Out of scope

- 階層ベイズの完全実装（別タスク）

## Deliverables

- `src/rxn_platform/tasks/assimilation.py` の拡張（Laplacian regularizer）
- `tests/test_assim_laplacian_regularizer_dummy.py`

## Acceptance Criteria

- lambda=0 で既存EKIと一致する
- lambda>0 でパラメータの滑らかさ指標が改善する（簡易チェック）

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

