# T026 Observable: gas_composition（指定種/集約）

- **id**: `T026`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T025
- **unblocks**: (none)
- **skills**: python, chemistry

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

最も基本的な観測量としてガス組成の要約を提供する。
多条件比較のため、同じ種集合・同じ集約方法で特徴量化できることが重要。

## Scope

- 指定種（もしくは上位N種）について、time series から last/mean/max/integral を計算
- 任意: 元素別（Si系, Cl系など）への集約もオプションで対応（elements情報があれば）
- 出力を ObservableArtifact の table に格納（列名規約: gas.<species>.<stat>）

## Out of scope

- 高速なスペクトル同化（後回し）

## Deliverables

- `src/rxn_platform/tasks/observables.py` に GasCompositionObservable を追加
- `tests/test_observable_gas_composition.py`

## Acceptance Criteria

- dummy run でも計算できる（species名はdummyの定義でOK）
- 列名が安定し、条件比較しやすい
- 設定で対象species/統計量が変えられる

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

