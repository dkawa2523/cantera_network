# T052G Phase Gate P2→P3: Optimization/Assimilation/Reduction/Golden/Benchmark の統合テスト・修正/軽リファクタ

- **id**: `T052G`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T040G + T041..T052（queue.json参照）
- **unblocks**: P3一式（T053..T060）
- **skills**: qa, integration, refactor

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`
- `docs/04_PIPELINES.md`
- `docs/06_TESTING_AND_QA.md`

## Background

P2では、最適化・データ同化・縮退と妥当性評価・回帰（golden）・ベンチマークが揃います。
ここは “実運用の中核” であり、
- 目的変数を差し替えても枠組みが壊れない
- 自動実行が途中で止まらず最後まで完走する
- 縮退の影響が自動で評価できる
ことが重要です。

そこで P2 完了時点で、
- dummy backend を中心に、P2機能を一通り通す統合テスト
- golden suite の回帰テストが安定して動く
- benchmark report まで生成できる（最小ケースで可）
を確認し、問題があればこのタスク内で修正/軽リファクタまで完了させます。

## Scope

### 統合テスト（必須）
- dummy backend で以下が最小構成で動くこと（小さな反復回数でOK）
  - optimization（random search）
  - assimilation（EKI もしくは ES-MDA のうち最低1つ）
  - reduction（threshold prune → validation loop）
- それぞれが Artifact 化され、manifest/parents が追跡できること

### 回帰テスト（必須）
- golden suite（P2で追加された回帰）が `pytest` で実行できること
- しきい値（許容差）が docs と一致すること（捏造禁止、根拠が無ければTODO）

### 修正/軽リファクタ（必要なら）
- ループが無限/過剰に重い場合は config で小さくできるよう調整
- 目的変数（observables）差し替え時の破綻を修正
- 縮退→再実行→差分評価の自動化が途中で止まる場合は、エラーを明確化し修正

## Out of scope

- P3の発展（Graph Laplacian, MBDoE, SBI, GNNなど）

## Deliverables

- （必要なら）`tests/test_p2_integration_dummy.py`（統合テスト）追加/調整
- （必要なら）`configs/pipeline/p2_smoke.yaml` 追加/調整
- （必要なら）バグ修正/軽リファクタ

## Acceptance Criteria

- P2 の Verification がすべて成功する
- dummy backend で P2 の統合テストが短時間で完走する（極端に遅くしない）
- golden suite が安定してパスする
- benchmark report（最小）まで生成できる

## Verification

```bash
$ pytest -q
$ python -m rxn_platform.cli pipeline run pipeline=p2_smoke sim=dummy
```

## Notes

- `pipeline=p2_smoke` が無い場合は、このタスク内で追加して良い（QA資産として残す）。
- ループ回数（budget/ensemble/iters）は config で小さくする（テストは小さく）。
