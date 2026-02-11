# T014 Pipeline runner v1: @id/@last参照・step timing・pipeline manifest出力

- **id**: `T014`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T013
- **unblocks**: (none)
- **skills**: python, pipeline

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/04_PIPELINES.md`

## Background

実運用では step の出力を `@id` で参照したい。
また、比較可能性のため pipeline 実行自体も manifest を残したい。

## Scope

- `@<step_id>` / `@last` 参照を inputs で解決できるようにする
- step timing（開始/終了/elapsed）を記録する
- pipeline実行結果を `PipelineRunArtifact`（kind=pipelines 等）として保存（manifest+results）

## Out of scope

- 完全なDAGスケジューラ（後回し）
- 外部workflow連携（後回し）

## Deliverables

- `src/rxn_platform/pipelines.py` の拡張
- `tests/test_pipeline_runner_refs.py`

## Acceptance Criteria

- @id/@last が正しく解決される
- pipeline run artifact が artifacts/ に保存される
- 同じ設定なら同じ pipeline_run_id になり再利用できる（可能なら）

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

