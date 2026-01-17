# T018 Report base: ReportArtifact（HTMLテンプレ）と共通ヘッダ/メタ埋め込み

- **id**: `T018`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T005, T012
- **unblocks**: (none)
- **skills**: python, reporting

## Contracts (must follow)

- `docs/05_VISUALIZATION_STANDARDS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

可視化は“考察の最短経路”だが、最初から作り込みすぎると破綻する。
まずは ReportArtifact の骨格（HTMLテンプレ + メタ埋め込み）を固定する。

## Scope

- `ReportArtifact`（kind=reports）として `index.html` を出力する枠組みを作る
- レポートに入力artifact id と生成日時、設定の要約を必ず表示する
- 図は空でも良いが、後続タスク（DS/Chem dashboard）が載る枠を確保

## Out of scope

- 本格的なWebアプリ化（後回し）

## Deliverables

- `src/rxn_platform/tasks/viz.py`（viz.base task など）
- `src/rxn_platform/reporting.py`（HTML生成ユーティリティ）
- `tests/test_report_artifact_smoke.py`

## Acceptance Criteria

- ReportArtifact が artifacts/reports/<id>/ に保存される
- index.html に manifest/config/入力artifact id が表示される
- vizタスクが再計算をしない（Artifactを読むだけ）

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

