# T040 Viz Chem dashboard v0: species/ROP/ネットワークsubgraph/縮退差分プレースホルダ

- **id**: `T040`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T018, T033, T035
- **unblocks**: (none)
- **skills**: python, plotly, reporting

## Contracts (must follow)

- `docs/05_VISUALIZATION_STANDARDS.md`

## Background

化学反応専門家観点の可視化（主要種時系列、ROPランキング、ネットワーク部分グラフ）をレポート化する。

## Scope

- species time series（上位Nまたは指定種）を表示
- ROP/wdotランキング（features または run から）を表示
- 反応ネットワークのサブグラフ（重要ノード/反応のみ）を可視化
- 縮退差分はプレースホルダでもよいが、差分入力があれば表示できるよう枠を用意

## Out of scope

- 高度なインタラクティブ探索（後回し）

## Deliverables

- `src/rxn_platform/tasks/viz.py` に chem_dashboard タスクを追加
- `tests/test_viz_chem_dashboard_smoke.py`

## Acceptance Criteria

- chem dashboard HTMLが生成される
- graph/ropが無い場合も落ちずに説明が出る

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

