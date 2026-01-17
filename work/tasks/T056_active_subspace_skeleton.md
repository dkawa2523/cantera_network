# T056 Active Subspace skeleton: 感度/サンプルから低次元方向を推定（簡易）

- **id**: `T056`
- **priority**: `P3`
- **status**: `todo`
- **depends_on**: T055
- **unblocks**: (none)
- **skills**: python, dimension-reduction

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

高次元パラメータ推定の効率化として Active Subspace は有力。
感度/サンプルから支配的方向を推定する最小実装を用意し、可視化/次元削減に繋げる。

## Scope

- サンプル点での勾配（感度）を集め、C = E[gg^T] を推定
- 固有分解で支配的固有ベクトル（active directions）を出力
- active subspace 上の可視化（2D scatterなど）は optional
- 結果を SubspaceArtifact として保存

## Out of scope

- 理論的な誤差評価（後回し）

## Deliverables

- `src/rxn_platform/tasks/dimred.py`（新規or features内）
- `tests/test_active_subspace_dummy.py`

## Acceptance Criteria

- 主成分方向が出力される
- 次元kをconfigで指定できる

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

