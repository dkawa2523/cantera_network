# T055 MBDoE/FIM skeleton: 感度から条件候補をランキング（D-opt等は将来）

- **id**: `T055`
- **priority**: `P3`
- **status**: `todo`
- **depends_on**: T037, T025
- **unblocks**: (none)
- **skills**: python, design-of-experiments

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

多条件実行を活かして、どの条件で観測すれば推定が良くなるか（実験計画）を支援したい。
ここでは感度から近似FIMを作り、条件候補をランキングする枠組みを用意する。

## Scope

- 候補条件セット（configリスト）を入力として受け取る
- 各条件での感度（T037等）から近似FIM（J^T W J）を計算（簡易）
- D-opt（logdet）等の指標で条件をランキングして出力
- 結果を DesignArtifact として保存（または table）

## Out of scope

- 厳密なMBDoE（後回し）

## Deliverables

- `src/rxn_platform/tasks/doe.py`（新規or optimization内）
- `tests/test_mbdoe_fim_dummy.py`

## Acceptance Criteria

- dummyでランキングが出る
- 条件数が増えても計算が破綻しない（簡易）

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

