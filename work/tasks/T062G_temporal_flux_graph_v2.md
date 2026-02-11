# Task: TemporalFluxGraph v2：時間窓・条件統合・疎行列保存を共通部品として確立（CNR/GNN/AMORE共通）

    ## Summary
    - **Goal**: 動的ネットワーク（時間窓フラックスグラフ）生成を“共通部品”にして、CNR/GNN/AMOREが同じ入力で動くようにする。
    - **Scope**: 時間窓切り（log/event/fixed）、ROP積分からspeciesグラフ生成、条件統合（mean/p95混合）、疎行列保存、キャッシュ。
    - **Non-goals**: 二部グラフ（species-reaction）完全対応はP1以降。まずspeciesグラフでMVP。

    ## Depends on
    - depends_on: T061G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/03_TEMPORAL_NETWORK_SPEC.md

    ## Acceptance Criteria
    - [ ] RunStoreに `graphs/species_graph/` として `layer_###.npz`（CSR）が保存される。
- [ ] windowingが `log_time` と `event_based` の少なくとも2種類で動く。
- [ ] 条件統合で `mean` と `p95` を併用でき、混合比がconfigで指定できる。
- [ ] 同一入力（mechanism+conditions+windowing）では再生成しない（cache hit）。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli run recipe=build_temporal_graph run_id=demo_graph`
- `python -m rxn_platform.cli show-graph run_id=demo_graph --layer 0`

    ## Notes
    - ノード順（species list）をmetaとして保存し、後続のmappingやGNN datasetで必ず一致させる。
- 行列は極端に密にならないよう、quantileでsparsifyオプションを用意する。
