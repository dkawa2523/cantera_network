# T030 Graphs v1: bipartite graph（species/reaction）を構築しJSON保存

- **id**: `T030`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T029
- **unblocks**: (none)
- **skills**: python, networkx

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

グラフアルゴリズムを適用するため、species/reaction の二部グラフ表現を構築し保存する。
ノード集約/経路削除/中心性などの解析が可能になる。

## Scope

- T029のS/metadataから bipartite graph を構築（networkx推奨、無ければ簡易dict）
- ノード: species_* と reaction_*
- エッジ: species→reaction（reactant/product, stoich係数符号付き）
- JSON（node-link）で保存し、可視化や他言語でも読みやすくする

## Out of scope

- 可視化の作り込み（T039/T040で）

## Deliverables

- `src/rxn_platform/tasks/graphs.py` に build_bipartite_graph を追加
- `tests/test_graph_bipartite_smoke.py`

## Acceptance Criteria

- graph JSON が保存される
- species/reactionの対応が追える（idが安定）

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

