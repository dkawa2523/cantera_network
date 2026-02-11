# T060 CI/Docs polish: 最小CI・README・開発者向けクイックスタート整備

- **id**: `T060`
- **priority**: `P3`
- **status**: `todo`
- **depends_on**: T017, T051, T019, T015
- **unblocks**: (none)
- **skills**: ci, docs

## Contracts (must follow)

- `docs/07_CONTRIBUTING.md`

## Background

最後に、開発者が迷わず導入できるよう docs/CI/README を整備する。
また、運用でコードがスパゲッティにならないためのガードレールを明文化する。

## Scope

- READMEに quick start（doctor, smoke pipeline, artifacts ls, viz）を追加
- CI（GitHub Actions等）が使えるなら最小のpytestを回す（使えない環境なら docs に手順のみ）
- docs/READMEの読み順と、拡張時のルール（新task追加の手順、contracts更新）を追記

## Out of scope

- 大規模なデプロイ自動化（後回し）

## Deliverables

- `README.md`（または docs/README.md）更新
- `.github/workflows/ci.yml`（任意）
- `docs/07_CONTRIBUTING.md` の追記

## Acceptance Criteria

- 新規開発者が README の手順で smoke まで到達できる
- CIがある場合、pytestが動く（短時間）

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

