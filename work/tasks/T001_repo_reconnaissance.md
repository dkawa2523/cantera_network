# T001 Repo reconnaissance: 既存Cantera実行例/出力/構造を調査し docs TODO を埋める

- **id**: `T001`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: (none)
- **unblocks**: (none)
- **skills**: repo-audit, docs

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/01_ARCHITECTURE.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

既存リポジトリ（Cantera自動実行テストコード/反応ネットワーク解析の既存断片）がどのような構造・入出力を持つかを先に把握しないと、
以後の実装で「既存資産を壊す」「出力形式が二重化する」「Cantera実行部が分離できない」などのリスクが高い。
本タスクでは、捏造せずに repo を走査して現状を記録し、docs の TODO を実データで埋める。

## Scope

- リポジトリ内の Cantera 実行スクリプト/設定ファイル/出力ログ（CSV/JSON/テキスト等）を探索し、場所と役割を整理する
- 現状の入出力（入力: mechanism/条件、出力: 濃度/ROP/ログ形式など）を `docs/08_REPO_MAP.md` として記録する
- `docs/*` 内の TODO（現状図/出力形式/依存/命名衝突など）を、観測できた範囲で更新する（不明点は TODO のまま）
- 既存コードと新基盤（rxn_platform）を共存させる方針（移動しない/ラップする/将来的に移行など）を短く明文化する

## Out of scope

- コードの大規模リファクタや移動（本タスクでは行わない）
- Cantera実行の再設計（後続タスクで実装）

## Deliverables

- `docs/08_REPO_MAP.md`（新規）: 既存資産の場所・役割・入出力・今後の取り扱い方針
- `docs/README.md` へ 08_REPO_MAP.md のリンクを追記
- 必要なら `work/EPICS.md` の“既存資産との接続点”を更新（任意）

## Acceptance Criteria

- `docs/08_REPO_MAP.md` に、少なくとも (1) 既存のCantera実行入口 (2) 既存出力形式 (3) 既存機構ファイルの扱い (4) 既存の自動実行方式 が記録されている
- docs の更新内容が捏造ではなく、実際にrepoに存在するファイル/構造に基づいている（不明点は TODO）
- 既存コードは一切壊さない（移動/削除をしない）

## Verification

```bash
$ python -c "print('T001 ok')"
```

## Notes

- 以後のタスクはこの `docs/08_REPO_MAP.md` を根拠に、既存コードとの接続（adapter）を最小差分で行う。

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

