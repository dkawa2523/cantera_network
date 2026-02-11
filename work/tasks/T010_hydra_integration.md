# T010 Hydra integration: config compose/override/--cfg表示 + 実行設定をmanifestへ保存

- **id**: `T010`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T009, T004
- **unblocks**: (none)
- **skills**: python, hydra

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/03_CONFIG_CONVENTIONS.md`

## Background

Hydraを前提に「設定が真実」を守るため、CLIからHydra configをcomposeして実行できるようにする。
また、実行時の有効設定を必ずmanifestに保存する基盤を作る。

## Scope

- hydra-core を用いて config group (sim/task/pipeline) をcomposeできるようにする
- `--cfg` 相当（解決済み設定の表示）をCLIに実装（例: `rxn cfg ...` or `--print-config`）
- seedを設定から受け取り、乱数を統一（numpy/python random）
- 実行した config dict を manifest.config.hydra に保存する（どのartifactでも）

## Out of scope

- 全てのコマンドでの完全なHydra対応（段階的に拡張）
- 複雑なhydra sweep最適化（後回し）

## Deliverables

- `configs/defaults.yaml`（最低限）
- `src/rxn_platform/hydra_utils.py`（compose/resolve/print）
- `tests/test_hydra_compose_smoke.py`（configが読める）

## Acceptance Criteria

- Hydra config をコードからcomposeできる
- 解決済み設定を表示できる
- manifest に config を埋め込むAPIが用意されている（Store/Taskで利用）

## Verification

```bash
$ python -m compileall -q src
```

## Notes

- Hydraが未導入の既存repoの場合は、このタスクで依存追加（pyproject）も行う。

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

