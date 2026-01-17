# T017 Doctor: 環境診断コマンド（dummy smoke含む）

- **id**: `T017`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T016, T009
- **unblocks**: (none)
- **skills**: python, qa

## Contracts (must follow)

- `docs/06_TESTING_AND_QA.md`

## Background

利用者/開発者が環境不備で止まらないよう、doctor を用意する。
CIが無い環境でも `rxn doctor` で最低限の健全性確認ができる。

## Scope

- CLIに `doctor` サブコマンドを実装
- 依存（hydra, numpy, xarray, cantera(optional)）の import チェック
- artifacts への書き込み可否チェック
- dummy smoke pipeline を実際に1回回す（短時間）

## Out of scope

- 重いベンチマークや本番機構の検証

## Deliverables

- `src/rxn_platform/doctor.py`（任意）または cli.py に実装
- `tests/test_doctor_smoke.py`（任意。CIで回す場合）

## Acceptance Criteria

- `python -m rxn_platform.cli doctor` が成功する（dummy前提）
- Canteraが無い場合でも doctor が分かりやすく警告しつつ続行できる

## Verification

```bash
$ python -m rxn_platform.cli doctor
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

