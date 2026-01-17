# T060G Phase Gate P3→Release: Advanced（Laplacian/MBDoE/AS/SBI/GNN/MF）を含む全体回帰・修正/整理

- **id**: `T060G`
- **priority**: `P3`
- **status**: `todo`
- **depends_on**: T052G + T053..T060（queue.json参照）
- **unblocks**: (none)
- **skills**: qa, release, refactor

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`
- `docs/06_TESTING_AND_QA.md`
- `docs/07_CONTRIBUTING.md`

## Background

P3は発展機能（Graph Laplacian, Laplacian正則化, MBDoE, Active Subspace, SBI, GNN dataset export, Multi-fidelity）であり、
運用上は **“無くても動くが、有ると強い”** 位置づけです。

この段階で重要なのは、
- optional 依存が無い環境でも基盤が壊れない（skip/graceful）
- P0〜P2のベースライン品質を落とさない
- 追加機能が artifact/manifest/parents の契約に沿う
ことです。

そこで Release 前の最終ゲートとして、全テスト/doctor/最小E2Eを実行し、
問題があればこのタスク内で修正・整理（軽リファクタ）まで完了させます。

## Scope

### 全体回帰（必須）
- `pytest -q` が通る
- `doctor` が通る（dummyベース）
- `p2_smoke` が通る（dummyベース）

### optional依存の扱い（必須）
- sbi 等が無い環境では tests が skip になる（failしない）
- 依存がある環境では最小の動作確認を行う

### 整理/軽リファクタ（必要なら）
- optional機能が baseline を壊す依存を持っていたら分離
- docs の TODO を埋められる範囲で整理（捏造禁止）

## Out of scope

- 本番大規模ベンチ（ここではやらない）

## Deliverables

- （必要なら）テストの skip 条件調整
- （必要なら）依存関係の分離/軽リファクタ
- （必要なら）README/CIの微修正

## Acceptance Criteria

- Verification が成功する
- optional機能が無い環境でも基盤が壊れない
- baseline（P0〜P2）を壊さない

## Verification

```bash
$ pytest -q
$ python -m rxn_platform.cli doctor
$ python -m rxn_platform.cli pipeline run pipeline=p2_smoke sim=dummy
```

## Notes

- `pipeline=p2_smoke` が無い場合は P2ゲート（T052G）で必ず作る。
