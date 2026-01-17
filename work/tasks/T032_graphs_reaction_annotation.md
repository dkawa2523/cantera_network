# T032 Graph annotation（reaction）: reaction_type/order/reversible 等を付与（unknown可）

- **id**: `T032`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T030
- **unblocks**: (none)
- **skills**: python, chemistry

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

縮退での反応集約には、反応種別（例: Elementary/Three-body/Surface/Adsorption など）の考慮が必要。
reactionノードに種別・次数等の属性を付与する。

## Scope

- Cantera reaction クラスから reaction_type を抽出（無ければルールベース）
- 反応次数の近似（reactantsの係数和）を付与
- reversible/duplicate などのフラグを付与
- 後続のreaction lumpingで使えるように meta を保存

## Out of scope

- 詳細な化学反応分類学の完全実装（段階的）

## Deliverables

- `src/rxn_platform/tasks/graphs.py` に annotate_reactions を追加
- `tests/test_reaction_annotation.py`

## Acceptance Criteria

- reaction_type が少なくとも unknown ではなく主要クラスに分類される（可能な範囲）
- 分類不能はunknown+理由が残る

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

