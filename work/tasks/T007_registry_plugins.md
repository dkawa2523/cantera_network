# T007 Registry: backend/task/observable/feature/viz のプラグイン登録・解決

- **id**: `T007`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T003, T005
- **unblocks**: (none)
- **skills**: python, plugin-system

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/01_ARCHITECTURE.md`

## Background

手法（Task/Observable/Backend/Feature）を後から追加してもスパゲッティ化しないために、
登録・解決を一箇所に集約する Registry を実装する。

## Scope

- Registry に `register(kind, name, obj)` / `get(kind, name)` / `list(kind)` を実装
- kind は最低限: backend, task, observable, feature, viz（必要に応じ追加）
- 重複登録時の挙動（上書き禁止 or 明示フラグ）を固定する
- 後で複数開発者が追加しやすいよう、登録は“1ファイル内に集約”しすぎない（ただしファイル増殖は抑える）

## Out of scope

- entrypoints 自動検出（後回し）
- 動的 import の複雑化

## Deliverables

- `src/rxn_platform/registry.py`
- `tests/test_registry_basic.py`

## Acceptance Criteria

- 登録→取得ができる
- 未知キーでの例外が分かりやすい
- 重複登録が静かに上書きされない（事故防止）

## Verification

```bash
$ python -m compileall -q src
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

