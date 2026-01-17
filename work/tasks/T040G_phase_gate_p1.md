# T040G Phase Gate P1→P2: Cantera/Graph/Feature/Sensitivity/Viz の統合テスト・修正/軽リファクタ

- **id**: `T040G`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T020G + T021..T040（queue.json参照）
- **unblocks**: P2一式（T041..T052）
- **skills**: qa, integration, refactor

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`
- `docs/04_PIPELINES.md`
- `docs/05_VISUALIZATION_STANDARDS.md`
- `docs/06_TESTING_AND_QA.md`

## Background

P1では、Cantera実行・反応ネットワーク化・特徴量化・感度解析・可視化が揃います。
ここが安定しないと、P2（最適化/同化/縮退）で
「どこが壊れているか分からない」状態になり、開発が遅延します。

そこで P1 完了時点で、
- dummy backend による “高速・再現可能” な統合テスト
- （可能なら）Cantera backend による “最小” の統合テスト
- Artifact契約（Run/Observable/Graph/Feature/Sensitivity/Report）の整合
を確認し、問題があればこのタスク内で修正・軽リファクタまで完了させます。

## Scope

### 統合テスト（必須）
- dummy backend で P1主要機能が E2E で動くこと
  - sim → observables → graphs → features → sensitivity → viz
- 生成された成果物に対して contract validators が通ること

### Cantera統合テスト（任意）
- Cantera が import 可能な環境なら、最小条件で同等の統合テストを 1 ケースだけ実施
- Cantera が無い環境では skip して良い（doctor/pytest に理由が出ること）

### 修正/軽リファクタ（必要なら）
- contract違反・I/O不整合・例外/ログ不備・キャッシュ破綻を修正
- コードの責務境界（カテゴリ独立）を守るための軽リファクタ

## Out of scope

- P2以降の最適化/同化/縮退ロジック
- heavy な Cantera ケースや大量条件 sweep

## Deliverables

- （必要なら）`tests/test_p1_integration_dummy.py` 等の統合テスト追加/調整
- （必要なら）`configs/pipeline/p1_smoke.yaml` の追加/調整（運用用）
- （必要なら）bugs / 小規模リファクタ

## Acceptance Criteria

- P1 の Verification がすべて成功する
- dummy backend の P1 E2E が 30秒程度で終わる（目安。極端に遅くしない）
- Cantera がある環境では Cantera 統合も 1 ケース通る
- 生成物が Artifact契約に準拠している（validatorが通る）

## Verification

```bash
$ pytest -q
$ python -m rxn_platform.cli pipeline run pipeline=p1_smoke sim=dummy
$ python - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec('cantera') is None:
    print('cantera not installed; skipping cantera integration')
    raise SystemExit(0)
subprocess.check_call([sys.executable, '-m', 'rxn_platform.cli', 'pipeline', 'run', 'pipeline=p1_smoke', 'sim=cantera_0d'])
print('cantera integration ok')
PY
```

## Notes

- `pipeline=p1_smoke` が未整備なら、このタスク内で追加して良い（QA資産として残す）。
- Canteraケースの mechanism は “Cantera同梱” の `gri30.yaml` 等を使ってよい（外部DL禁止）。
