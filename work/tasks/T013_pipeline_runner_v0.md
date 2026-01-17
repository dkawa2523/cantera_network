# T013 Pipeline runner v0: steps YAML を逐次実行しArtifactをつなぐ（最小）

- **id**: `T013`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T012
- **unblocks**: (none)
- **skills**: python, pipeline

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/04_PIPELINES.md`

## Background

YAML宣言で step を並べて実行できると、多条件実行・同化・縮退評価のフローが“再利用可能”になる。
まずは逐次実行で良いので pipeline runner を作る。

## Scope

- `configs/pipeline/*.yaml` 形式の steps を読み込む（最小スキーマ）
- steps を上から順に Task runner で実行し、出力artifact idをstep結果として保持
- step間の受け渡しは `inputs` に artifact id を明示する形でまず成立させる（@参照はT014）

## Out of scope

- DAG並列実行（後回し）
- 高度な再開（後回し）

## Deliverables

- `src/rxn_platform/pipelines.py`（PipelineRunner）
- `tests/test_pipeline_runner_v0.py`（dummy stepを2つ実行）

## Acceptance Criteria

- pipeline yaml で2step以上が動く（dummy）
- 失敗したstepが分かる（例外メッセージ/ログ）
- stepの出力artifact idが集約される（dict等）

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

