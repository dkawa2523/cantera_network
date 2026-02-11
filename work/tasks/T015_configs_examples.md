# T015 Configs: dummy sim + smoke pipeline + task configs のサンプル追加

- **id**: `T015`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T014, T011
- **unblocks**: (none)
- **skills**: yaml, hydra

## Contracts (must follow)

- `docs/03_CONFIG_CONVENTIONS.md`

## Background

運用を迷わせないために、最小の “動く設定” を同梱する。
dummy backend で smoke pipeline が通ることが、以後の回帰の基準になる。

## Scope

- `configs/defaults.yaml` の defaults chain を整備
- `configs/sim/dummy.yaml`（dummy backend）
- `configs/pipeline/smoke.yaml`（sim.run を1回回す最小pipeline）
- `configs/task/*.yaml`（simタスク等の雛形）

## Out of scope

- 本番用の膨大な装置設定（後回し）

## Deliverables

- `configs/defaults.yaml`
- `configs/sim/dummy.yaml`
- `configs/pipeline/smoke.yaml`

## Acceptance Criteria

- `pipeline=smoke sim=dummy` で最小実行ができる前提が整う
- 設定ファイルが docs の規約に沿う（group名、overrideなど）

## Verification

```bash
$ python -c "from pathlib import Path; assert Path('configs/pipeline/smoke.yaml').exists(); print('configs ok')"
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

