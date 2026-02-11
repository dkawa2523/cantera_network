# Task: GNN-B：動的GNNで反応/種重要度を推定→QoI条件付きPruningで縮約メカを合成（閾値探索付き）

    ## Summary
    - **Goal**: 動的GNNで反応/種重要度を推定し、QoI条件付きで縮約メカ（YAML）を合成して運用可能にする。
    - **Scope**: 弱教師（ROP積分/QoI寄与）+自己教師、重要度出力、閾値探索、MechanismCompilerでYAML生成、val検証。
    - **Non-goals**: AMOREのような完全探索。ここは“軽量で安定な縮約メカ生成”を狙う。

    ## Depends on
    - depends_on: T066G, T061G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/02_REDUCTION_METHODS.md

    ## Acceptance Criteria
    - [ ] 重要度が time-window × case で出力され、集約（max/weighted）で keep_S/keep_R を作れる。
- [ ] 縮約YAML（reduced_mech.yaml）が生成され、Canteraで実行できる。
- [ ] 閾値探索により、QoI許容内の最小サイズメカが選択される。
- [ ] 重要度ヒートマップ等の可視化が1つ以上出力される。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=reduce_gnn_importance_prune run_id=demo_gnn_prune`
- `python -m rxn_platform.cli run recipe=validate_mech run_id=demo_gnn_prune`

    ## Notes
    - 教師は最初は '明らかに重要/不要' のみでも良い。残りは自己教師で埋める。
- MechanismCompilerはAMOREと共通化し重複禁止。
