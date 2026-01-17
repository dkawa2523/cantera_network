# T039 Viz DS dashboard v0: 条件分布/目的変数/感度ヒートマップ/収束プレースホルダ

- **id**: `T039`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T018, T025, T034, T037
- **unblocks**: (none)
- **skills**: python, plotly, reporting

## Contracts (must follow)

- `docs/05_VISUALIZATION_STANDARDS.md`

## Background

データサイエンティスト観点の可視化（分布、相関、感度、最適化履歴）をレポート化する。
“分析に必要な全て”を一度に作るのではなく、拡張可能な枠組みで実装する。

## Scope

- ReportArtifactに Plotly（推奨）等で図を埋め込む（静的HTML）
- 少なくとも: 条件分布、目的変数の散布図/ヒスト、感度ヒートマップ を表示
- 入力artifact（runs/observables/sensitivity）をmanifestに記録
- 図の選択/対象列はconfigで変更可能にする

## Out of scope

- UI操作性の作り込み（後回し）

## Deliverables

- `src/rxn_platform/tasks/viz.py` に ds_dashboard タスクを追加
- `tests/test_viz_ds_dashboard_smoke.py`（最小）

## Acceptance Criteria

- レポートHTMLが生成され、ブラウザで開ける
- 入力が欠けていても、欠けた旨を表示して落ちない

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

