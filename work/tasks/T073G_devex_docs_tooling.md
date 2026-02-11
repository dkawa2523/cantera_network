# Task: 開発者体験の改善：ドキュメント整備（Quickstart/How-to add reducer）＋lint/format/type導入

    ## Summary
    - **Goal**: 他開発者が迷わず追従できるよう、DXとドキュメント、静的解析を整備する。
    - **Scope**: Quickstart、Add new reducer、Coding conventions、pre-commit、ruff/black/mypy導入（軽量）。
    - **Non-goals**: 完全な型付け100%。重要モジュールから段階導入。

    ## Depends on
    - depends_on: T060G, T072G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/06_REFACTORING_POLICY.md

    ## Acceptance Criteria
    - [ ] docs/README.md が“読む順”になり、最短の実行手順が1ページで分かる。
- [ ] 新しい縮退法追加の手順が docs に明記される（Plugin registry前提）。
- [ ] lint/format/typeの最低限がCI/ローカルで回る。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli help`
- `ruff check . || true`

    ## Notes
    - “ファイル数を増やしすぎない”を優先し、共通処理は core/ に寄せる。
