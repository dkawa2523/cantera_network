# T047 Reduction v1: threshold prune（importance/sensitivity 기반）で反応削除候補生成

- **id**: `T047`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T046, T038
- **unblocks**: (none)
- **skills**: python, chemistry

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

縮退のベースラインとして、感度/重要度に基づく threshold prune を実装する。
重要でない反応を無効化する候補を自動生成し、patchとして出力する。

## Scope

- SensitivityArtifact（またはfeatures）から反応重要度スコアを取得
- 閾値（topK/score<th）で disabled_reactions を決め、patchを生成
- 反応種別（reaction_type）に基づく保護ルール（例: surface反応は残す等）をオプションで追加

## Out of scope

- 最適な閾値探索（後回し）

## Deliverables

- `src/rxn_platform/tasks/reduction.py` に threshold_prune を追加
- `tests/test_reduction_threshold_prune.py`

## Acceptance Criteria

- 入力importanceに応じてpatchが生成される
- 保護ルールがconfigで指定できる

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

