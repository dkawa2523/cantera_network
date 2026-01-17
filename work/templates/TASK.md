# <TASK_ID> <TITLE>

- **id**: `<TASK_ID>`
- **priority**: `<P0|P1|P2|P3>`
- **status**: `todo`
- **depends_on**: (none)  <!-- use queue.json for the source of truth -->
- **unblocks**: (none)
- **skills**: (optional; e.g., python, cantera, graph, qa)

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/01_ARCHITECTURE.md`
- `docs/02_ARTIFACT_CONTRACTS.md`
- `docs/03_CONFIG_CONVENTIONS.md`
- `docs/04_PIPELINES.md`
- `docs/05_VISUALIZATION_STANDARDS.md`
- `docs/06_TESTING_AND_QA.md`
- `docs/07_CONTRIBUTING.md`

## Background
（なぜ必要か、現状課題）

## Scope
- 

## Out of scope
- 

## Deliverables
- [ ] (file / module / config)

## Acceptance Criteria
- [ ] (observable behavior)

## Verification
以下のコマンドがローカルで成功すること（可能なら network 不要）。

```bash
$ <command>
```

## Notes
- 迷う点・根拠が無い点は TODO（捏造禁止）
- 追加依存は最小。optional import / importorskip を活用してテストを壊さない
- ファイル増殖を避け、カテゴリ単位で責務をまとめる（ただし巨大1ファイル化もしない）

## Final Response (Codex)
Codex の最終応答は **必ず** tools/codex_loop/response_schema.json の JSON に従う。

