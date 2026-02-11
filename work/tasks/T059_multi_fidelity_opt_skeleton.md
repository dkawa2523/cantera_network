# T059 Multi-fidelity optimization skeleton: 縮退機構を低忠実度として探索に利用

- **id**: `T059`
- **priority**: `P3`
- **status**: `todo`
- **depends_on**: T041, T046, T048
- **unblocks**: (none)
- **skills**: python, optimization

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

縮退機構を低忠実度モデルとして探索に利用すると、最適化コストを下げられる可能性がある。
ここでは multi-fidelity optimization の枠（full vs reduced）だけ実装する。

## Scope

- full model と reduced model（patch適用）の2種類の評価関数を用意
- 探索ではまず low-fidelity を多用し、上位候補を high-fidelity で再評価する（簡易）
- 結果（どの忠実度で何回評価したか）をartifactに残す

## Out of scope

- 厳密なMFBO（後回し）

## Deliverables

- `src/rxn_platform/tasks/optimization.py` の拡張（multi_fidelity mode）
- `tests/test_multi_fidelity_opt_dummy.py`

## Acceptance Criteria

- dummyでmulti-fidelityフローが動く
- 高忠実度評価回数がconfigで制御できる

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

