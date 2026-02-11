# Task: GNN-C：候補評価サロゲートでAMORE探索を加速（active learning＋不確実性ゲート）

    ## Summary
    - **Goal**: AMORE探索の候補評価回数を減らし、探索を現実速度にする（active learning）。
    - **Scope**: 候補メカ特徴（削除率、反応タイプ比、動的フラックス保持率）→誤差/失敗確率予測、上位のみ厳密評価、オンライン更新。
    - **Non-goals**: 最先端不確実性推定の網羅。まず“外れ候補を弾く”安全運用を優先。

    ## Depends on
    - depends_on: T064G, T066G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md

    ## Acceptance Criteria
    - [ ] サロゲートモデルが候補メカの誤差と失敗確率を予測できる（学習ログ保存）。
- [ ] AMORE探索でサロゲートにより評価回数が削減される（before/afterをレポート）。
- [ ] 不確実性が高い候補は厳密評価へ回すゲートがある（安全側）。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=reduce_amore_search --use_surrogate true run_id=demo_amore_surr`

    ## Notes
    - 最初はGBDTでも良いが、GNNにするなら二部グラフ表現を検討（後で）。
- active learningのデータはRunStoreに 'datasets/surrogate/' として保存し再利用可能に。
