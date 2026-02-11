# Task: CNR-Coarse拡張：動的ネットワーク（時間窓）×制約付きクラスタで mapping 作成＋superstate特徴量出力

    ## Summary
    - **Goal**: CNR-Coarseのコア成果物（mapping）を動的ネットワークから生成し、superstate特徴量を標準Artifactとして出力する。
    - **Scope**: 制約分割→各グループで重み付きコミュニティ検出→mapping統合、superstate射影（X_super, wdot_super）、可視化。
    - **Non-goals**: coarse ODE実行は別タスク。まず解析/同化に効くmappingと特徴量を固める。

    ## Depends on
    - depends_on: T062G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/02_REDUCTION_METHODS.md
- docs/03_TEMPORAL_NETWORK_SPEC.md

    ## Acceptance Criteria
    - [ ] RunStoreに `reduction/cnr_coarse/mapping.json` が保存される（phase/charge等で混ぜない）。
- [ ] 各caseの `features/superstate_timeseries.zarr` が生成される（X_super, wdot_super 等）。
- [ ] mappingの品質指標（flux coverage, cluster size stats）が `metrics.json` に入る。
- [ ] 可視化（cluster graph / sankeyなど）を `viz/` に1つ以上出す。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=reduce_cnr_coarse run_id=demo_cnr`
- `python -m rxn_platform.cli report run_id=demo_cnr`

    ## Notes
    - コミュニティ検出はまずLeiden/Louvain（重み付き）でMVP。multiplexは後でもよい。
- hard constraintsは“事前分割”で担保し、後付け制約最適化は避ける。
