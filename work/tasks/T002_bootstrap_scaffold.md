# T002 Bootstrap scaffold: package骨格・artifacts/work/configs/tests最小整備

- **id**: `T002`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T001
- **unblocks**: (none)
- **skills**: python, repo-scaffolding

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/01_ARCHITECTURE.md`

## Background

P0として、以後の全タスクが“迷わず”作業できる最小の足場（package / configs / artifacts / tests）を整える。
既存リポジトリを壊さずに共存できるよう、追加は最小にする。

## Scope

- Python package（`src/rxn_platform/`）の最小骨格を追加
- `configs/`, `artifacts/`, `tests/`, `work/` の存在を保証（既にある場合は衝突回避）
- 最低限の `pyproject.toml`（または既存があれば追記）を整備し、importできる状態にする
- 成果物/ログをgitに入れない `.gitignore` を整備（既存がある場合は追記）

## Out of scope

- 依存関係の大量追加（必要最小限に留める）
- 既存コードの移動・削除

## Deliverables

- `src/rxn_platform/__init__.py`
- `src/rxn_platform/cli.py`（空でも良い。後タスクで実装）
- `configs/.gitkeep`（必要なら）
- `artifacts/.gitkeep`（必要なら）
- `tests/test_import_smoke.py`（import確認）
- `.gitignore`（artifacts/, work/logs/, work/state.json 等を除外）

## Acceptance Criteria

- `python -c "import rxn_platform"` が成功する
- `artifacts/` と `configs/` と `tests/` が存在する（空でも可）
- 既存の実行例が壊れていない（既存の入口ファイルは触らない）

## Verification

```bash
$ python -c "import rxn_platform; print('import ok')"
$ python -m compileall -q src
```

## Notes

- パッケージ名は衝突があれば変更可。ただし docs と configs と一致させること。

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

