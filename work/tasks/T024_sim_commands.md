# T024 sim コマンド: init/validate/run/viz（backend独立）+ キャッシュskip

- **id**: `T024`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T023, T009, T015
- **unblocks**: (none)
- **skills**: python, cli, hydra

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/03_CONFIG_CONVENTIONS.md`

## Background

backend独立な `sim` コマンド群を実装し、入力編集→実行→出力→簡易可視化までを一貫で回せるようにする。
同時に「将来他シミュレータへ差し替え」できるよう、backendIFに依存して実装する。

## Scope

- `sim init` : テンプレconfigを生成（既存configsをコピーでも可）
- `sim validate` : mechanismや必須パラメータの存在確認（軽量）
- `sim run` : backend選択して実行し RunArtifact を保存（キャッシュskipあり）
- `sim viz` : RunArtifactの簡易プロット（T/P/主要種）をReportArtifactで出力（最小）

## Out of scope

- 高度なGUI（後回し）
- 並列sweepコマンド（後回し。pipeline/optimizerで代替）

## Deliverables

- `src/rxn_platform/cli.py` simサブコマンド実装
- `src/rxn_platform/tasks/sim.py` の拡張（viz用の最小可視化）
- `tests/test_sim_cli_smoke.py`（dummyでOK）

## Acceptance Criteria

- dummy backend で `sim run` が動く
- cacheが効き、同run_idが既にある場合は再計算しない（ログに明示）
- validate が失敗理由を出す

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

