# T058 GNN dataset export: 動的グラフ/イベント列をArtifactとして出力（自己教師前処理）

- **id**: `T058`
- **priority**: `P3`
- **status**: `todo`
- **depends_on**: T033, T022
- **unblocks**: (none)
- **skills**: python, graph-ml

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

GNN/自己教師あり学習のためには、複数条件のシミュレーション結果から“動的グラフデータセット”を作る必要がある。
ここでは dynamic graph / event sequence を抽出してArtifact化する。

## Scope

- RunArtifact（time series）と GraphArtifact（構造）から、時刻ごとのノード/エッジ特徴量を抽出
- 最小: time×species のXやwdotを node feature として出力
- データセット分割（train/val/test）は後でも良いが、run_idごとのメタ情報を残す
- 出力は GnnDatasetArtifact（parquet/npz）として保存

## Out of scope

- GNNモデル学習そのもの（別途タスク）

## Deliverables

- `src/rxn_platform/tasks/gnn_dataset.py`（新規）
- `tests/test_gnn_dataset_export_dummy.py`

## Acceptance Criteria

- dummy run + dummy graph でdataset artifactが生成できる
- ノード順序や対応が明確に保存される

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

