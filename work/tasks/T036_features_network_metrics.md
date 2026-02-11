# T036 Features: graph/network metrics + 多条件安定性（rank stability等）

- **id**: `T036`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T034, T033
- **unblocks**: (none)
- **skills**: python, graph-analysis

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

ネットワーク指標（中心性など）を特徴量として同化/縮退に使えるようにする。
また複数条件での安定性（重要度が条件で変わるか）を評価する。

## Scope

- GraphArtifactから中心性等を計算し features に取り込む
- 複数条件（複数run）でのrank stability（Spearman等）を計算してmetaに保存
- 条件間で重要ノード/反応が一貫するかを示す指標を出す

## Out of scope

- 高度な統計検定（後回し）

## Deliverables

- `src/rxn_platform/tasks/features.py` に NetworkMetricFeature を追加
- `tests/test_features_network_metrics.py`

## Acceptance Criteria

- GraphArtifactがあればfeaturesが出る
- 複数runを与えると安定性指標が計算される

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

