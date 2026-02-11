# T046 Reduction v0: mechanism patch schema（disable reaction / multiplier）+ apply

- **id**: `T046`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T029, T020
- **unblocks**: (none)
- **skills**: python, yaml, chemistry

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

縮退（反応削除/集約）を管理するには、操作を“差分（patch）”として表現するのが安全。
本タスクで patch スキーマと apply を実装する。

## Scope

- patch形式（YAML/JSON）で disabled_reactions / reaction_multipliers を表現
- apply_patch: mechanism YAML を読み込み、反応リストをフィルタして新mechanism YAMLを生成（speciesは残してよい）
- apply結果のmechanism pathを sim config に差し替えられるようにする
- patch自体も Artifact として保存できるようにする（kind=patches等）

## Out of scope

- species削除の完全対応（後回し）
- lumpingの適用（後続タスク）

## Deliverables

- `src/rxn_platform/tasks/reduction.py`（Patch schema + apply）
- `tests/test_reduction_patch_apply.py`（小さなYAMLでOK）

## Acceptance Criteria

- patch→new YAML が生成される
- 適用操作が可逆（元mechanismを壊さない）
- 適用したpatchがmanifestに記録される

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

