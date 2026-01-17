# T031 Graph annotation（species）: formula/elements/charge/state/phase を付与

- **id**: `T031`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T030
- **unblocks**: (none)
- **skills**: python, chemistry

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

縮退でのノード集約には、化合物の状態（ラジカル/イオン/中性）や元素組成の類似性が必要。
本タスクで species ノードに化学的属性を付与する。

## Scope

- speciesごとに: 元素組成（dict）、電荷、相（gas/surface/solid）を付与
- 状態分類: ion（charge!=0）, neutral, radical（heuristic/unknown可）
- 化学式や元素組成ベクトルをmetaとして保存し、後続のnode lumpingで利用できるようにする

## Out of scope

- 完璧なラジカル判定（TODO可）

## Deliverables

- `src/rxn_platform/tasks/graphs.py` に annotate_species を追加
- `tests/test_species_annotation.py`

## Acceptance Criteria

- 主要な属性がgraphに入る
- 未知/推定の場合は flag が立つ（is_inferredなど）

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

