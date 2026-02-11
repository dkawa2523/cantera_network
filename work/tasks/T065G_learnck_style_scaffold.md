# Task: LearnCK-Styleスキャフォールド：stable states spec＋射影データセット＋overall reaction テンプレ＋検証ループ

    ## Summary
    - **Goal**: LearnCK-Styleの最小運用形を用意し、stable-only縮約の足場を作る（本格学習は後でも回る）。
    - **Scope**: stable states spec、full runからstableへの射影データ生成、overall reactionテンプレ（保存則を満たす構造）、val検証ループ。
    - **Non-goals**: 高度なNN学習の最適化・性能追求。まず動くパイプライン＋評価を優先。

    ## Depends on
    - depends_on: T061G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/02_REDUCTION_METHODS.md

    ## Acceptance Criteria
    - [ ] stable states を `reduction/learnck/stable_states.yaml` として保存し、差し替え可能にする。
- [ ] RunStoreに stable射影データ（timeseries/feature）が保存される。
- [ ] overall reaction 構造テンプレが生成され、保存則（元素/サイト）チェックが通る。
- [ ] val条件でQoI誤差レポートが出る（未達の場合は TODO として明記）。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=reduce_learnck_style run_id=demo_learnck`
- `python -m rxn_platform.cli report run_id=demo_learnck`

    ## Notes
    - 表面サイト制約（占有率0..1、総サイト一定）を最初からハードに入れる。
- stable選定は最初は手動+重要度（ROP/QoI寄与）で十分。自動化はP3以降。
