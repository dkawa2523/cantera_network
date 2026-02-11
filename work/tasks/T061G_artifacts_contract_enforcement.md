# Task: Artifact契約の強制と移行：RunStore manifest/metrics/config の必須化＋legacy互換読み込み

    ## Summary
    - **Goal**: すべてのパイプラインが同一のArtifact契約で連結できるようにし、legacy出力も読み取り可能にして移行を止めない。
    - **Scope**: manifest/metrics/configの必須化、RunStore APIの強制、legacy読み込み（read-only）アダプタ、破損run検出（doctor）。
    - **Non-goals**: 過去runの全面移行（変換）は必須にしない。読み取り互換でOK。

    ## Depends on
    - depends_on: T060G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/05_ARTIFACT_CONTRACTS.md

    ## Acceptance Criteria
    - [ ] RunStoreの各runに `manifest.json`, `config_resolved.yaml`, `metrics.json` が必ず生成される。
- [ ] doctorが“契約違反run”を検出し、修復手順（再生成/再実行）を案内できる。
- [ ] legacy出力（旧ディレクトリ）を `rxn_platform.cli import-legacy ...` でRunStoreに参照登録できる（実体コピーは任意）。

    ## Verification
    Commands to run (examples):
    - `python -m rxn_platform.cli doctor --strict`
- `python -m rxn_platform.cli run recipe=sim_sweep run_id=demo_contract`

    ## Notes
    - 後続タスクはすべてRunStoreから読み、RunStoreへ書く。途中だけ別形式は禁止。
- manifestには `simulator`, `mechanism_hash`, `conditions_hash`, `qoi_spec_hash` を入れて再現性を担保。
