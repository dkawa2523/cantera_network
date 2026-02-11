# Task: AMORE-Search拡張：MechanismCompiler共通化＋キャッシュ＋cheap dynamic score で探索を現実速度へ

    ## Summary
    - **Goal**: AMORE探索を“回る速度”に落とし、縮約メカ（YAML）生成を運用可能にする。
    - **Scope**: MechanismCompiler共通化、候補hash+キャッシュ、cheap dynamic score（重要フラックス保持率）で事前ふるい、beam探索。
    - **Non-goals**: MCTSなど高度探索は後。まずbeam+cacheで安定運用。

    ## Depends on
    - depends_on: T062G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/02_REDUCTION_METHODS.md

    ## Acceptance Criteria
    - [ ] 候補メカが `mechanism_hash` でキャッシュされ、同一候補の再評価をしない。
- [ ] cheap scoreが temporal graph の統計から算出でき、低スコア候補を評価前に除外できる。
- [ ] 探索結果として `reduction/amore/reduced_mech.yaml` と `edit_log.json` と `pareto.csv` が保存される。
- [ ] val条件で合格する最小サイズメカが1つ選ばれ、QoI誤差がレポートされる。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=reduce_amore_search run_id=demo_amore`
- `python -m rxn_platform.cli diff-mech run_id=demo_amore`

    ## Notes
    - メカ編集（remove/merge）と評価（simulate+QoI）は完全分離してテスト可能にする。
- 探索が止まらないよう、エラー候補は 'failed' として記録し次へ進む。
