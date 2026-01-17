# T051 Golden suite: 代表条件セット + 回帰しきい値 + pytest回帰テスト

- **id**: `T051`
- **priority**: `P2`
- **status**: `todo`
- **depends_on**: T016, T048
- **unblocks**: (none)
- **skills**: python, pytest, qa

## Contracts (must follow)

- `docs/06_TESTING_AND_QA.md`

## Background

開発が進むほど“いつ壊れたか”の特定が難しくなる。
代表条件の golden suite を定義し、許容誤差付きで回帰テスト化する。

## Scope

- dummyベースの代表条件セット（速い）を定義し、期待されるObservable/Feature範囲を保存
- pytestで golden suite を実行し、閾値超過で落ちるようにする
- Cantera版は optional（canteraがあれば追加で回す）

## Out of scope

- 重いベンチ（後回し）

## Deliverables

- `tests/test_golden_suite_dummy.py`
- `configs/golden/*.yaml`（任意）
- docs/06_TESTING_AND_QA.md への追記（任意）

## Acceptance Criteria

- golden suite がCI/ローカルで回る（短時間）
- 閾値がconfigで変更できる

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

