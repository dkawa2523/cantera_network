# T048 Reduction v2: validation loop（縮退→再実行→差分評価）とValidationArtifact

- **id**: `T048`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T047, T024, T025, T034
- **unblocks**: (none)
- **skills**: python, qa, simulation-orchestration

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/06_TESTING_AND_QA.md`

## Background

縮退は“結果が変わらない”ことの検証が核心。
縮退patchを順次適用→自動実行→目的変数差分を評価する validation loop を実装する。

## Scope

- baseline run と reduced run を同条件で実行（pipeline利用）
- Observable/Feature の差分を計算し、許容誤差内なら pass とする
- 縮退レベル（複数patch候補）を順次評価し、最も縮退したpass案を選べるようにする（v0は逐次でOK）
- 結果を ValidationArtifact として保存（pass/fail, metrics, selected patch）

## Out of scope

- 高速な探索（後回し）

## Deliverables

- `src/rxn_platform/tasks/reduction.py` に validate_reduction を追加
- `tests/test_reduction_validation_dummy.py`

## Acceptance Criteria

- dummyで validation loop が動く
- pass/fail が明確に出力される
- 差分指標/許容誤差がconfigで変更できる

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

