# T020 Contract validators: Run/Observable/Graph/Feature/Sensitivityの整合チェック関数

- **id**: `T020`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T003, T005
- **unblocks**: (none)
- **skills**: python, validation

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

カテゴリ独立を守るため、Artifactの最低限の整合チェックを共通化する。
後続タスク（backend/observables/graphs/features/sensitivity）で“契約違反”を早期検出できる。

## Scope

- RunArtifact / ObservableArtifact / GraphArtifact / FeatureArtifact / SensitivityArtifact の validate 関数を用意
- 必須フィールド（manifestのkind/schema_version 等、dataの基本列/次元）を検査
- 不足している場合は ValidationError を投げる（エラーメッセージに不足項目を列挙）

## Out of scope

- 完全なスキーマ検証（将来拡張）

## Deliverables

- `src/rxn_platform/validators.py`（validate_* 関数群）
- `tests/test_validators_smoke.py`（dummy artifactで検査）

## Acceptance Criteria

- 契約違反の原因が分かる（どの変数/列が無いか）
- validator を Task runner / doctor から呼べる（少なくとも import できる）

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

