# T004 Core IDs: config正規化・hash・run_id/artifact_idの安定生成

- **id**: `T004`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T003
- **unblocks**: (none)
- **skills**: python, hashing

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/03_CONFIG_CONVENTIONS.md`

## Background

run_id / artifact_id の安定性は「比較可能性」「キャッシュ再利用」の根幹。
Hydra設定を元に、順序に依存しない正規化とハッシュ生成を実装する。

## Scope

- Hydraの設定dictを安定に正規化（キー順、list/tuple差、numpy型など）する関数を実装
- 正規化した設定から `stable_hash` を生成し、run_id/artifact_id に使えるようにする
- ハッシュ生成から除外するキー（ログ設定等）を明確化できる仕組みを用意（configで指定可、または関数引数）
- 同じ設定→同じID、設定が違う→違うID となることをテストで保証

## Out of scope

- artifact保存そのもの（T005）
- 高度な差分表示（後回し）

## Deliverables

- `src/rxn_platform/core.py` に `canonicalize(obj)` / `stable_hash(obj)` / `make_run_id(cfg)` / `make_artifact_id(...)`
- `tests/test_stable_hash.py`

## Acceptance Criteria

- 辞書の順序が違っても同じhashになる
- 数値型（int/float/numpy）やPathが混じってもhashが安定する（可能な範囲）
- 除外キーの指定ができる（最小でOK）

## Verification

```bash
$ python -m compileall -q src
$ python -c "from rxn_platform.core import stable_hash; print(stable_hash({'b':2,'a':1}), stable_hash({'a':1,'b':2}))"
```

## Notes

- このID仕様は docs にも反映する（必要なら `docs/02_ARTIFACT_CONTRACTS.md` に追記）。

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

