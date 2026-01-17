# T043 Assimilation v0: パラメータ化（selected reactions）+ prior sampling + misfit/weights

- **id**: `T043`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T025, T037, T020
- **unblocks**: (none)
- **skills**: python, data-assimilation

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

同化では「どのパラメータを推定するか」「観測誤差をどう扱うか」が最重要。
本タスクでパラメータ化・事前分布・ミスフィット計算の枠を固定する。

## Scope

- 推定対象パラメータ（例: reaction multipliersのsubset）を定義し、ParameterVectorとして扱う
- 事前分布（例: log-normal, bounded uniform）からサンプルするAPIを用意
- 観測値（膜厚/組成/占有率など複数）と予測値（Observable）から misfit を計算する（重み/ノイズ）
- 結果を AssimilationArtifact（または PosteriorArtifact）へ保存できる形にする

## Out of scope

- EKI/ES-MDA更新（T044/T045で実施）

## Deliverables

- `src/rxn_platform/tasks/assimilation.py`（Parameterization + misfit）
- `tests/test_assim_param_misfit.py`

## Acceptance Criteria

- パラメータのサンプリングが再現可能
- misfitがベクトル/スカラー両方で計算できる
- 観測対象を後から差し替えやすい（列名で指定）

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

