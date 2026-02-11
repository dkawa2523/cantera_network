# T042 Optimization v1: 多目的ベースライン（pareto archive / 簡易NSGA-II）

- **id**: `T042`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T041
- **unblocks**: (none)
- **skills**: python, optimization

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

多目的（膜厚・組成・占有率など）を扱うため、Paretoベースのベンチマークを用意する。
外部ライブラリ無しで成立する最小の Pareto archive / 簡易NSGA-II を実装する。

## Scope

- 目的関数ベクトルを扱えるようにする（ObservableArtifactから複数列）
- 非支配ソート（Pareto archive）を実装し、best集合を保持
- 任意: 簡易進化（mutationのみ）で改善を試みる（NSGA-II完全実装でなくてよい）
- 結果を OptimizationArtifactに保存

## Out of scope

- 高速な大規模進化計算（後回し）

## Deliverables

- `src/rxn_platform/tasks/optimization.py` の拡張
- `tests/test_opt_pareto_archive.py`

## Acceptance Criteria

- 目的関数が複数でも破綻しない
- Pareto front が出力される

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

