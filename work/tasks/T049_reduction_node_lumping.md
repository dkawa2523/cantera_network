# T049 Reduction v3: node lumping prototype（元素/状態/電荷類似）+ mapping artifact

- **id**: `T049`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T046, T031
- **unblocks**: (none)
- **skills**: python, graph, chemistry

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

ノード集約（species lumping）は状態/元素組成の類似性を考慮すべき。
ただしmechanismへの適用は難しいため、まず候補生成と評価指標をartifact化する。

## Scope

- species annotation（T031）を使い、元素組成ベクトル+電荷+相+状態で類似度を定義
- クラスタリング（簡易: hierarchical/threshold）で代表ノードを決め、mappingを出力
- mappingを NodeLumpingArtifact として保存
- 将来applyするため、代表種の選択根拠（頻度/中心性など）もmetaに残す

## Out of scope

- mechanismファイルの書き換え（後回し）

## Deliverables

- `src/rxn_platform/tasks/reduction.py` に propose_node_lumping を追加
- `tests/test_node_lumping_proposal.py`

## Acceptance Criteria

- 類似度とmappingが出力される
- 状態/元素組成/電荷が考慮されている

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

