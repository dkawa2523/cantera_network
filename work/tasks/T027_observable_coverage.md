# T027 Observable: coverage_summary（表面被覆率。無ければgraceful）

- **id**: `T027`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T025
- **unblocks**: (none)
- **skills**: python, chemistry

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

ALD/CVDでは表面被覆率（coverage）が重要な目的変数/中間特徴量になる。
ただしRunArtifactにcoverageが無い場合も多いので、gracefulに扱う。

## Scope

- RunArtifactに coverage 変数（time×surface_species）があれば要約統計を作る
- coverageが無い場合は ObservableArtifact に NaN と理由（missing_coverage）を残す
- 列名規約: cov.<species>.<stat>

## Out of scope

- 表面反応モデルの自動推定（後回し）

## Deliverables

- `src/rxn_platform/tasks/observables.py` に CoverageObservable を追加
- `tests/test_observable_coverage_missing_ok.py`

## Acceptance Criteria

- coverage無しRunArtifactで落ちない
- coverage有りの場合に値が出る（簡単なダミーでも可）

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

