# Task: GNNデータ基盤：TemporalFluxGraph→PyG dataset 変換＋optional extras（gnn）＋遅延import対応

    ## Summary
    - **Goal**: GNN系の学習/推論に必要なDatasetとFeatureをRunStoreから生成できるようにする（依存はoptional）。
    - **Scope**: TemporalFluxGraph（layers）→PyTorch Geometric Data 変換、条件/case分割、メタ（species list）整合、extras[gnn]。
    - **Non-goals**: モデル精度の最大化。まず再現性あるdataset生成と保存。

    ## Depends on
    - depends_on: T062G, T060G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/03_TEMPORAL_NETWORK_SPEC.md

    ## Acceptance Criteria
    - [ ] `pip install .[gnn]` でのみGNN依存が入る（標準インストールは軽いまま）。
- [ ] DatasetBuilderがRunStoreから `datasets/temporal_graph_pyg/` を生成できる（train/val split付き）。
- [ ] 依存が無い環境でもCLIは壊れない（遅延import/明確なエラーメッセージ）。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=build_gnn_dataset run_id=demo_gnn_ds`

    ## Notes
    - Datasetのキーは (case_id, window_id) を基本にし、再現性のために順序・seedを固定する。
- ノード順（species）や制約グループのIDをメタに含める。
