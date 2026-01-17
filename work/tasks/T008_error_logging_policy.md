# T008 Error/Logging policy: 例外階層・ユーザ向けエラー・ログ出力規約を実装

- **id**: `T008`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T003
- **unblocks**: (none)
- **skills**: python, logging

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/07_CONTRIBUTING.md`

## Background

自動ループ運用では、失敗理由が分からないと止まり続ける。
例外階層とログ規約を用意し、どのカテゴリでも同じ“落ち方”をするようにする。

## Scope

- 基底例外 `RxnPlatformError` と派生（ConfigError/ArtifactError/BackendError/ValidationError）を定義
- CLIやTask runnerで例外を捕捉し、ユーザ向けに短いメッセージ+詳細ログを出す方針を実装
- logging設定（最低: logger名、レベル、フォーマット）を1箇所に用意

## Out of scope

- リッチなログUI（後回し）
- 分散トレーシング

## Deliverables

- `src/rxn_platform/errors.py`
- `src/rxn_platform/logging_utils.py`（任意）
- `tests/test_error_messages.py`（例外が握り潰されないこと）

## Acceptance Criteria

- 代表的な失敗（missing config / missing artifact / missing backend）で、原因が分かる例外が出る
- stacktraceの全文表示はオプションにできる（CLIフラグ or ログレベル）

## Verification

```bash
$ python -m compileall -q src
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

