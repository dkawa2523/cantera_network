# T033 Graph analytics: SCC/communities/階層メトリクス算出（FeatureArtifactでも可）

- **id**: `T033`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T031, T032
- **unblocks**: (none)
- **skills**: python, graph-analysis

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

ネットワークの階層性・循環性を解析するために、SCCやコミュニティ等のグラフ解析を実装する。
結果は features や reduction の重要度に利用される。

## Scope

- SCC（強連結成分）を計算し、循環構造を抽出
- コミュニティ検出（簡易: connected components / modularityベース optional）
- 中心性（degree/betweenness など）を計算し重要ノードをランキング
- 結果を GraphArtifactまたはFeatureArtifactとして保存

## Out of scope

- 最高性能のコミュニティ法（後回し）

## Deliverables

- `src/rxn_platform/tasks/graphs.py` に analyze_graph を追加
- `tests/test_graph_analytics_smoke.py`

## Acceptance Criteria

- 少なくともSCCとdegree中心性が計算できる
- 大規模でも落ちないよう、計算対象（上位N等）をconfigで制限できる

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

