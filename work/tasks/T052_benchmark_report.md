# T052 Benchmark report: baseline vs advanced（最適化/同化/縮退）比較レポート生成

- **id**: `T052`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T041, T044, T048, T039, T040, T051
- **unblocks**: (none)
- **skills**: python, reporting

## Contracts (must follow)

- `docs/06_TESTING_AND_QA.md`

## Background

手法の有効性を比較するには、ベンチマークレポートが必要。
同一の artifact 群から baseline（random/threshold prune）と advanced（EKI等）を比較するレポートを生成する。

## Scope

- OptimizationArtifact / AssimilationArtifact / ValidationArtifact などを入力に取り、集約表を作る
- 主要指標（目的関数、誤差、縮退率、計算回数/時間）をまとめる
- ReportArtifact としてHTML出力し、再現可能にする（入力artifact idをmanifestに記録）

## Out of scope

- 学術レベルの詳細評価（後回し）

## Deliverables

- `src/rxn_platform/tasks/viz.py` または `reporting.py` に benchmark_report を追加
- `tests/test_benchmark_report_smoke.py`

## Acceptance Criteria

- dummy artifact群でもレポート生成が動く
- 入力不足の場合も落ちずに警告表示される

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

