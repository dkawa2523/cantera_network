# T034 Features framework + timeseries summary（mean/max/integral/last）

- **id**: `T034`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T012, T020
- **unblocks**: (none)
- **skills**: python, pandas, xarray

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

多目的/多条件の同化・最適化では、時系列をそのまま使うより“要約特徴量”が必要になる。
FeatureExtractorの枠組みを作り、timeseriesの基本統計をfeaturesとして出力する。

## Scope

- FeatureExtractor base を定義し registry に登録できるようにする
- RunArtifactから指定変数（T/P/主要種など）の mean/max/min/last/integral を計算
- FeatureArtifact として table を保存（行=run, 列=feature）

## Out of scope

- 学習モデル構築（このタスクではしない）

## Deliverables

- `src/rxn_platform/tasks/features.py`（framework + TimeseriesSummaryFeature）
- `tests/test_features_timeseries.py`

## Acceptance Criteria

- dummy run でもfeaturesが出る
- 欠損変数はNaNで扱い、理由をmetaに残す

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

