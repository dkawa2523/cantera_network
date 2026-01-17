# T053 Graph Laplacian: L/D/正規化ラプラシアン計算 + 保存（GraphArtifact拡張or別Artifact）

- **id**: `T053`
- **priority**: `P3`
- **status**: `todo`
- **depends_on**: T033
- **unblocks**: (none)
- **skills**: python, graph, linear-algebra

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

グラフラプラシアンは、GNNや正則化、階層ベイズの事前構造など幅広く使える。
ここではネットワークから Laplacian を計算し再利用可能な形で保存する。

## Scope

- Graph（species graph または similarity graph）から degree matrix D と Laplacian L = D - A を計算
- 正規化ラプラシアン（optional）も計算可能にする
- L を保存（npz等）し、対応するノード順序を明記する
- 後続の正則化/学習で使えるユーティリティ関数を提供

## Out of scope

- 巨大グラフ向け最適化（後回し）

## Deliverables

- `src/rxn_platform/tasks/graphs.py` に build_laplacian を追加
- `tests/test_graph_laplacian.py`

## Acceptance Criteria

- Lが対称/半正定値など基本性質を満たす（簡単なチェック）
- ノード順序が保存される

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

