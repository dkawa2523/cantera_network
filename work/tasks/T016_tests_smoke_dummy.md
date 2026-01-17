# T016 Tests: dummy smoke pipeline integration test（pytest）

- **id**: `T016`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T015
- **unblocks**: (none)
- **skills**: python, pytest

## Contracts (must follow)

- `docs/06_TESTING_AND_QA.md`

## Background

自動ループで壊れないためには、最小の統合テストが必須。
dummy backend + smoke pipeline を `pytest` で固定する。

## Scope

- pytest を前提に tests を整備（既存テストがある場合は共存）
- dummy backend で pipeline smoke が成功し、artifacts に run が生成されることを検証
- テストはネットワーク不要・短時間で終わること

## Out of scope

- Cantera を使う重いテスト（後回し/optional）

## Deliverables

- `tests/test_smoke_pipeline_dummy.py`
- 必要なら `pyproject.toml` に dev 依存（pytest）追記

## Acceptance Criteria

- `pytest -q` が通る（dummy smoke）
- 失敗した場合に原因が分かる（assertメッセージ）

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

