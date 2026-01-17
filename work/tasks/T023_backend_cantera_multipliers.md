# T023 Cantera parameterization: reaction multipliers（摂動/無効化）適用とmanifest記録

- **id**: `T023`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T022
- **unblocks**: (none)
- **skills**: python, cantera

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

反応係数最適化/同化/感度では「反応係数（倍率）を独立に振る」が基本操作。
Canteraの reaction multipliers を統一的に適用し、manifestに残す。

## Scope

- config で指定された reaction multipliers（index or reaction_id）をCanteraに適用する
- disabled_reactions（multiplier=0）も同じ仕組みで表現可能にする
- 適用した multipliers を RunArtifact.manifest.inputs または provenance に保存
- 同じ multipliers の再実行で run_id が変化する（ID計算に含める）

## Out of scope

- ArrheniusパラメータA/Eaなどの直接推定（将来）

## Deliverables

- `src/rxn_platform/backends/cantera.py` multipliers適用ロジック
- `src/rxn_platform/core.py`（IDに multipliers を含める補助）
- `tests/test_multiplier_id_changes.py`（dummyで可）

## Acceptance Criteria

- multipliersを変えると run_id が変わる
- 同じ multipliers なら同じ run_id でキャッシュ再利用できる
- 適用した multipliers が manifest に記録される

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

