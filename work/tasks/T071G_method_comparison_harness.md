# Task: 統合比較ハーネス：同一ベンチ（GRI30 325等）で CNR/AMORE/LearnCK/GNN を回しレポート生成

    ## Summary
    - **Goal**: 同一ベンチ条件で複数縮退法を回し、比較可能なレポートを1コマンドで出す。
    - **Scope**: ベンチ定義（GRI30 325等）、各reducer実行、QoI誤差/速度/サイズの比較表、図、結論サマリ。
    - **Non-goals**: 論文品質の統計解析。まず意思決定に足る比較。

    ## Depends on
    - depends_on: T063G, T064G, T065G, T068G, T069G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md

    ## Acceptance Criteria
    - [ ] 1コマンドで指定したreducer群を順に実行し、同一指標で比較できる。
- [ ] 比較結果が `runs/<exp>/<run_id>/reports/comparison.md` などに保存される。
- [ ] 各手法の設定（recipe）と結果がリンクされ再現可能。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=benchmark_compare run_id=demo_compare`

    ## Notes
    - 比較は同一train/val split を強制し、skewを防止（docs/00_INVARIANTS）。
