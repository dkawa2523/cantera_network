# T012 Task runner: registryからTaskを解決し単体実行できるrunnerを実装

- **id**: `T012`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T007, T005, T010
- **unblocks**: (none)
- **skills**: python, task-runner

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/04_PIPELINES.md`

## Background

カテゴリ独立の実行単位を Task に統一する。
`task runner` があれば、Codexは“task単位で完結”した実装を繰り返せる。

## Scope

- Task基底（run(store, cfg)）を定義
- registry から task 名を解決して実行する runner を実装
- CLI の `task run` から task を起動できるようにする（最低限）
- Taskの入出力は Artifact だけ、という不変条件を徹底（docs参照）

## Out of scope

- pipeline runner（T013〜）
- 個別task（observables等）の実装（後続）

## Deliverables

- `src/rxn_platform/tasks/base.py`（Task base / TaskContext）
- `src/rxn_platform/tasks/runner.py`（または pipelines内）
- CLI: `rxn task run task=<name> ...` の最小実装
- `tests/test_task_runner_dummy.py`

## Acceptance Criteria

- registry登録されたtaskがCLIから実行できる（dummyでOK）
- taskの失敗がRxnPlatformError系で返る
- taskの実行結果artifact idがログ/標準出力で分かる

## Verification

```bash
$ python -m rxn_platform.cli task --help
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

