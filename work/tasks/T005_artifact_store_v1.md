# T005 ArtifactStore v1: layout/atomic write/read/immutability enforcement

- **id**: `T005`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T004
- **unblocks**: (none)
- **skills**: python, artifacts

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

ArtifactStore はカテゴリ独立（疎結合）を成立させるための唯一のデータ結合点。
不変条件として「イミュータブル」「原子書き込み」「比較可能性」を守れる実装にする。

## Scope

- `artifacts/<kind>/<id>/` のディレクトリレイアウトで保存/読込する
- manifest.yaml を必ず保存し、data（zarr/parquet/jsonなど）も同一フォルダに置く
- atomic write（tmp→rename）で中途半端生成物を残しにくくする
- 既に同じidが存在する場合は再生成しない（上書き禁止、必要なら新ID）

## Out of scope

- 高速検索用の全体index（T006で補助）
- 分散ストレージ対応

## Deliverables

- `src/rxn_platform/store.py`（ArtifactStore実装）
- `tests/test_artifact_store_roundtrip.py`（manifest保存/復元、簡単なdataファイル保存）

## Acceptance Criteria

- ArtifactStoreで manifest の write/read ができる
- 同じ id の artifact が既にある場合に上書きしない（例外 or skip）
- パス解決（kind/id）と存在確認が1箇所に集約されている

## Verification

```bash
$ python -m compileall -q src
$ python -c "from rxn_platform.store import ArtifactStore; s=ArtifactStore('artifacts'); print(s.root)"
```

## Notes

- 後続タスクで xarray(zarr)/parquet/json の保存ヘルパを追加していく。

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

