# T019 Artifacts CLI: artifacts ls/show/export（開発者が成果物を追える）

- **id**: `T019`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T009, T005
- **unblocks**: (none)
- **skills**: python, cli

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

成果物が増えると“どれが正か分からない”問題が起きる。
開発者が artifact を一覧/参照できるCLIを用意する。

## Scope

- CLIに `artifacts ls`（kind別一覧）を追加
- `artifacts show <kind> <id>`（manifest表示）を追加
- 任意: `artifacts open-report <id>` はURL出力だけでも良い（ブラウザ起動は不要）

## Out of scope

- 高度な検索UI（後回し）

## Deliverables

- `src/rxn_platform/cli.py` の artifacts サブコマンド実装
- `tests/test_artifacts_cli.py`（最小）

## Acceptance Criteria

- artifacts ディレクトリの内容を壊さずに一覧/参照できる
- 存在しないidを指定したときのエラーが分かりやすい

## Verification

```bash
$ python -m rxn_platform.cli artifacts --help
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

