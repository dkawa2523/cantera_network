# T037 Sensitivity v0: multiplier finite-diff で target（Observable）感度を算出

- **id**: `T037`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T023, T025, T014, T020
- **unblocks**: (none)
- **skills**: python, simulation-orchestration

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

反応係数最適化/同化の効率化には、どの反応が目的変数に効くかの感度が重要。
ここでは multiplier の有限差分で感度を計算し、SensitivityArtifactとして保存する。

## Scope

- 対象反応集合（reaction indices/ids）を入力として受け取る
- ベースラインrunと摂動run（multiplier*(1+eps)）を自動実行（pipelineを利用）
- 目的変数（Observable）の差分から感度を計算（dY/dlogk など、定義はconfig）
- キャッシュ再利用（同一run_idは再実行しない）

## Out of scope

- 随伴感度（後回し）

## Deliverables

- `src/rxn_platform/tasks/sensitivity.py`（SensitivityTask）
- `tests/test_sensitivity_fd_dummy.py`（dummy backend + dummy observable）
- `configs/task/sensitivity_fd.yaml`（雛形）

## Acceptance Criteria

- dummyで感度計算が動く（高速）
- 結果がテーブルとして保存され、反応ランキングが得られる
- epsや感度定義がconfigで切替できる

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

