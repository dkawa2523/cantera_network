# T035 Features: ROP/wdot summary（Top-N, time-integral）

- **id**: `T035`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T034, T022
- **unblocks**: (none)
- **skills**: python, chemistry

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

化学反応の重要度を示す典型的特徴量として ROP や wdot の要約が必要。
上位Nの反応/種を抽出し、特徴量として安定に出力する。

## Scope

- RunArtifactに rop/wdot がある場合に time-integral / max を計算
- 上位Nを抽出してfeatures化（列名に reaction_id/species を含める）
- rop/wdotが無い場合はgracefulにskip（NaN）

## Out of scope

- ROPの厳密分解/経路分解（将来）

## Deliverables

- `src/rxn_platform/tasks/features.py` に RopWdotFeature を追加
- `tests/test_features_rop_optional.py`

## Acceptance Criteria

- rop/wdotがあるRunArtifactで特徴量が出る
- 無い場合に落ちない

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

