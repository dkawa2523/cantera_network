# Task: ClearML 連携の下地（実装は軽量）：RunStore→ClearML task 変換アダプタ（optional dependency）

    ## Summary
    - **Goal**: 将来のClearMLタスク化に備え、RunStoreメタデータをClearMLへ同期できる最小アダプタを用意する（オプション）。
    - **Scope**: clearml extra、RunStore manifest→ClearML task fields、無依存環境ではnoop。
    - **Non-goals**: ClearMLの全機能活用。まずはタスク/アーティファクト登録の最小。

    ## Depends on
    - depends_on: T061G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/07_CLEARML_FUTURE.md

    ## Acceptance Criteria
    - [ ] `pip install .[clearml]` のときのみ連携が有効になる。
- [ ] run完了時にmanifest/metricsをClearML taskとして登録できる（dry-runモードも）。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=sim_sweep run_id=demo_clearml --clearml true --dry_run true`

    ## Notes
    - ClearMLは“将来想定”であり、基盤を複雑化させないこと。
