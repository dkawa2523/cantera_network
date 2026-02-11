# Task: 設定/出力の簡素化 v2：yaml爆発と出力階層を解消し RunStore を唯一の真実にする

    ## Summary
    - **Goal**: yaml過多・出力階層過多を解消し、ユーザーが迷わず実行/結果確認できる最小構成へ整理する。
    - **Scope**: Hydra設定の縮退（default+recipes）、RunStoreへの出力一本化、CLIの実行導線の統一、既存設定/出力の互換導線の用意。
    - **Non-goals**: 縮退アルゴリズム自体の精度改善（本タスクは土台整備）。ClearMLやDataset管理の本格実装。

    ## Depends on
    - depends_on: (none)

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/06_REFACTORING_POLICY.md

    ## Acceptance Criteria
    - [ ] configs/ は `default.yaml` + `recipes/*.yaml` を中心に再構成され、groupの多段ネストが撤去される（残す場合は理由をdocsに記載）。
- [ ] 実行結果の保存先が RunStore (`runs/<exp>/<run_id>/`) に統一され、深い階層や散逸した出力が発生しない。
- [ ] CLIは `python -m rxn_platform.cli run recipe=... run_id=...` を基本とし、実行コマンドがdocsに明記される。
- [ ] 既存の複雑なyaml/出力からの移行手順（または互換読み込み）を `docs/04_CONFIG_SIMPLIFICATION.md` に追記する。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli doctor`
- `python -m rxn_platform.cli run recipe=sim_sweep run_id=demo_t060g`
- `python -m rxn_platform.cli list-runs --last 5`

    ## Notes
    - “設定が真実”はStructured Config（dataclass）側に寄せ、YAMLは短いレシピに限定する。
- Hydraの `outputs/` や `multirun/` を一次保存先にしない（RunStoreに寄せる）。
- 後続タスク（GNN/AMORE等）の入出力がRunStore依存になるため、ここで破綻させない。
