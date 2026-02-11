# T045 Assimilation v2: ES-MDA baseline（任意/オプション依存）

- **id**: `T045`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T044
- **unblocks**: (none)
- **skills**: python, data-assimilation

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

ES-MDAは観測を複数回に分けて更新することで、非線形でも安定しやすい。
EKIとコード共有しつつ、オプションとして実装する。

## Scope

- ES-MDA の更新（alphaスケジュール）を実装
- EKIと同じパラメータ化/ミスフィットを利用
- 依存追加無しで実装（線形代数はnumpy）
- 計算負荷が高いのでdummyでのテストを中心にする

## Out of scope

- 高度なadaptive alpha（後回し）

## Deliverables

- `src/rxn_platform/tasks/assimilation.py` に ES-MDA 実装
- `tests/test_assim_esmda_dummy.py`

## Acceptance Criteria

- dummyでES-MDAが動く
- 更新回数やalphaがconfigで制御できる

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

