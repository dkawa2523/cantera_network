# Task: GNN-A：動的GNN（TGAT/TGN相当）＋DiffPoolで学習可能な状態マージ（mapping）を生成

    ## Summary
    - **Goal**: 動的GNN＋Poolingで学習可能なstate merge（mapping）を生成し、CNRでは拾いにくい条件依存構造を吸う。
    - **Scope**: TGAT/TGN相当のtemporal encoder（まず簡易でOK）、DiffPoolでassignment S学習、hard constraint分割、mapping出力。
    - **Non-goals**: NeuralODE置換。GNNでODEを解くのは禁止（Canteraを使い続ける）。

    ## Depends on
    - depends_on: T066G, T061G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/02_REDUCTION_METHODS.md
- docs/03_TEMPORAL_NETWORK_SPEC.md

    ## Acceptance Criteria
    - [ ] 学習済みmappingが `reduction/gnn_pool/mapping.json` としてRunStoreに保存される。
- [ ] hard constraints（phase/charge/formula等）で混ぜないことが構造的に保証される。
- [ ] 自己教師損失（edge weight再構成/次窓予測など）が最低1つ実装される。
- [ ] val条件で mapping の最低限評価指標（flux coverage）がレポートされる。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=reduce_gnn_pool_temporal run_id=demo_gnn_pool`

    ## Notes
    - 最初はsnapshot列（windowごと）で動くTGAT風でも良い。イベントTGNはP2後半で拡張。
- DiffPoolのcluster数Kはスイープして最小合格を選ぶ設計にする。
