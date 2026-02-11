# Task: データセット管理の下地：条件テーブル＋時系列データのdataset manifest/registry（将来拡張用）

    ## Summary
    - **Goal**: 条件テーブル＋時系列データを“データセット”として扱うためのmanifest/registryを用意する（将来拡張用）。
    - **Scope**: dataset manifest（conditions hash, mech hash, schema version）、登録/参照CLI、RunStoreとリンク。
    - **Non-goals**: DVC等の本格導入。まず軽量なregistry。

    ## Depends on
    - depends_on: T061G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/07_CLEARML_FUTURE.md

    ## Acceptance Criteria
    - [ ] dataset manifest が `datasets/registry.json` などに保存され、runから参照できる。
- [ ] 同一条件/メカで重複生成を防ぐ（hash一致なら再利用）。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli dataset list`
- `python -m rxn_platform.cli dataset register --run_id demo_graph`

    ## Notes
    - ClearMLや将来の外部ストレージに移行可能な抽象に留める。
