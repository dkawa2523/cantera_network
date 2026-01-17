# T050 Reduction v4: reaction lumping prototype（reaction_type内クラスタ）

- **id**: `T050`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T046, T032
- **unblocks**: (none)
- **skills**: python, graph, chemistry

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

反応集約（reaction lumping）は反応種別を考慮しつつ、似た反応をまとめる必要がある。
まずは reaction_type 内でのクラスタ候補生成を実装する。

## Scope

- reaction annotation（T032）を使い、reaction_typeごとに候補を分割
- 反応の反応物/生成物集合の類似度（Jaccard等）でクラスタリングしてmappingを出力
- mappingを ReactionLumpingArtifact として保存
- 適用は後回し（候補生成まで）

## Out of scope

- mechanism適用（後回し）

## Deliverables

- `src/rxn_platform/tasks/reduction.py` に propose_reaction_lumping を追加
- `tests/test_reaction_lumping_proposal.py`

## Acceptance Criteria

- reaction_type を考慮した候補が出る
- 類似度定義がconfigで変更できる

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

