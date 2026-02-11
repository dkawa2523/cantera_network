# T029 Graphs v0: 機構からS行列（species×reaction）を作りGraphArtifact保存

- **id**: `T029`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T021, T020
- **unblocks**: (none)
- **skills**: python, sparse-matrix, chemistry

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

反応ネットワーク解析の出発点として、機構から化学量論行列S（species×reaction）を構築して保存する。
これは階層性、中心性、Laplacian等の基礎データになる。

## Scope

- mechanism（Cantera Solution）から species list と reaction list を取得
- reactant/product stoichiometric coefficients から S を構築（scipy sparse 推奨、無ければnumpy）
- S と対応する metadata（species名、reaction equation、reaction_id）を GraphArtifact として保存

## Out of scope

- 階層クラスタリング/縮退（後続タスク）

## Deliverables

- `src/rxn_platform/tasks/graphs.py` に build_stoich 関数/Task を追加
- `tests/test_graph_stoich_dummy.py`（dummy機構でも可。canteraあればsmall）

## Acceptance Criteria

- S の形状と対応metadataが一貫している
- cantera無し環境ではテストskipまたはdummyで成立する

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

