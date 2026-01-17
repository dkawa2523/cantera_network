# T006 ArtifactStore v2: provenance/parents/再利用キャッシュ補助API

- **id**: `T006`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T005
- **unblocks**: (none)
- **skills**: python, artifacts

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

大量実行で重要なのは「既に計算済みなら再利用する」こと。
ArtifactStore にキャッシュ再利用の補助API（get-or-create / manifest比較 / parents記録）を追加する。

## Scope

- Artifactの存在確認 + 再利用（skip）の統一APIを用意（例: `store.ensure(kind, id, writer_fn)`）
- parents/inputs を manifest に入れて provenance を追えるようにする補助関数を追加
- 任意: `artifacts/_index.jsonl` など軽量な追記式インデックス（parquetは後でも可）

## Out of scope

- 高度なDB検索（後回し）
- 巨大indexの最適化

## Deliverables

- `src/rxn_platform/store.py` に `exists()/open_manifest()/write_artifact_dir_atomic()` 等を追加
- `tests/test_store_cache_semantics.py`（同idの上書き禁止/再利用）

## Acceptance Criteria

- 同じartifact_idで二重生成しない（再利用される）
- parents/inputs を manifest に入れるためのユーティリティがある
- キャッシュ再利用の挙動がテストで固定されている

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

