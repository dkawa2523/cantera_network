# Task: GNN-A評価：mapping品質指標（flux coverage / QoI近傍保持 / 安定性）＋CNRとの比較レポート自動生成

    ## Summary
    - **Goal**: GNN mapping の品質を“数値と図”で説明できるようにし、CNRとの差を比較可能にする。
    - **Scope**: flux coverage / QoI近傍窓保持 / cluster安定性 / サイズ分布、比較レポート生成、可視化。
    - **Non-goals**: 論文レベルの評価指標網羅。まず運用に必要な指標を固定。

    ## Depends on
    - depends_on: T067G, T063G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md

    ## Acceptance Criteria
    - [ ] mapping評価メトリクスが `metrics.json` に保存される（最低3指標）。
- [ ] CNR mapping と GNN mapping の比較レポートが自動生成される（markdownまたはhtml）。
- [ ] 少なくとも1つの図（cluster size, sankey, coverage over time等）が `viz/` に出る。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=eval_mapping run_id=demo_gnn_pool --compare demo_cnr`

    ## Notes
    - QoI近傍窓の定義はQoIプラグインに寄せ、ここで特定用途に固定しない。
