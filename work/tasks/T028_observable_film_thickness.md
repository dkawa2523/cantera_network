# T028 Observable: film_thickness v0（deposition_rate積分/簡易モデル）

- **id**: `T028`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T025, T022
- **unblocks**: (none)
- **skills**: python, domain-chemistry

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

観測対象が膜厚（膜圧/膜厚）である場合、気相状態から“表面反応を通した膜成長”を推定する必要がある。
本タスクではまず v0 として簡易モデルで膜厚を計算し、後から差し替えやすい構造にする。

## Scope

- RunArtifactに deposition_rate（time）または film_growth_flux 相当があれば、それを積分して膜厚を算出
- 無い場合は設定で proxy（例: 特定種の消費量）を指定できるようにし、最低限の推定を可能にする
- 推定根拠（どの変数から計算したか）を meta に残す
- 列名規約: film.thickness

## Out of scope

- 高精度な表面反応モデリング（将来タスク）

## Deliverables

- `src/rxn_platform/tasks/observables.py` に FilmThicknessObservable を追加
- `tests/test_observable_film_thickness_proxy.py`（dummyで可）
- `configs/observable/film_thickness.yaml`（雛形）

## Acceptance Criteria

- 最低限の推定ができ、壊れずに計算できる
- 推定根拠が meta に残る
- 後から別実装に置き換えられる（Observableとして独立）

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

