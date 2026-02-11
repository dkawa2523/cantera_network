# T009 CLI v0: sim/task/pipeline/viz/doctor の枠だけ作り --help を固定

- **id**: `T009`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T007, T008
- **unblocks**: (none)
- **skills**: python, cli

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/03_CONFIG_CONVENTIONS.md`

## Background

運用上の入口を固定する。CLIが安定していれば、内部実装を変えてもユーザが迷わない。
ここでは “枠” と “helpの安定” を最優先し、実処理は後タスクで埋める。

## Scope

- argparse（推奨）で CLI 骨格を実装（依存追加を避ける）
- サブコマンド群を追加: sim / task / pipeline / viz / doctor / artifacts
- 各サブコマンドは `--help` が出る（未実装でもエラーが分かる）

## Out of scope

- 全機能の実装（後続タスクで実装）
- 補完やUIの作り込み

## Deliverables

- `src/rxn_platform/cli.py`（main関数 + argparse構成）
- `tests/test_cli_help.py`（helpが出る）

## Acceptance Criteria

- `python -m rxn_platform.cli --help` が動く
- サブコマンド `sim/task/pipeline/viz/doctor/artifacts --help` が動く
- 未実装部分は NotImplementedError ではなく、ユーザに行動が分かるエラーにする

## Verification

```bash
$ python -m rxn_platform.cli --help
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

