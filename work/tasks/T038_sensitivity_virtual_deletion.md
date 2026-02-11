# T038 Sensitivity v1: virtual deletion（multiplier=0）+ キャッシュ + 条件間安定性

- **id**: `T038`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T037
- **unblocks**: (none)
- **skills**: python, simulation-orchestration

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

縮退や重要反応抽出では virtual deletion（反応を無効化）が有効。
また多条件で重要度が安定かどうかも評価する必要がある。

## Scope

- multiplier=0 の run を自動実行し、目的変数差分を計算（impact）
- 結果を sensitivity と同じフォーマットで保存（impact列など）
- 条件間安定性（重要反応が条件で変わらないか）を評価しmetaに保存
- キャッシュ再利用を徹底

## Out of scope

- 大規模探索の高速化（後回し）

## Deliverables

- `src/rxn_platform/tasks/sensitivity.py` の拡張
- `tests/test_sensitivity_virtual_deletion.py`

## Acceptance Criteria

- virtual deletion が動き、ランキングが出る
- 条件間安定性指標が出る

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

