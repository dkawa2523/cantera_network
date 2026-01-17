# T044 Assimilation v1: EKI baseline（pipeline rerun loop）+ PosteriorArtifact保存

- **id**: `T044`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T043, T014, T024
- **unblocks**: (none)
- **skills**: python, data-assimilation, linear-algebra

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

EnKF系（EKI）は“モデルをブラックボックス”として同化でき、Cantera自動実行と相性が良い。
ベンチマークとしてEKIを実装し、結果をartifact化する。

## Scope

- Ensemble Kalman Inversion (EKI) の更新式を実装（基本形でOK）
- 各イテレーションで ensemble の各メンバを pipeline 実行して予測Observableを得る
- 更新後のパラメータ ensemble と misfit 推移を保存（PosteriorArtifact）
- 数値安定のための正則化（ridge、inflation等）をオプションで追加

## Out of scope

- 非線形高度手法（後回し）

## Deliverables

- `src/rxn_platform/tasks/assimilation.py` に EKI 実装
- `tests/test_assim_eki_dummy.py`

## Acceptance Criteria

- dummy backend で EKI が短時間で回る
- 反復ごとのmisfitが保存される
- 途中で失敗した場合もログ/manifestで状況が追える

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

