# Task: P0〜P2の回帰テスト拡充：各recipeのsmoke/integration test＋再現性（seed）＋失敗診断

    ## Summary
    - **Goal**: 実装が増えても壊れないよう、回帰テストと失敗診断を整備する。
    - **Scope**: recipeごとのsmoke test、主要reducerのintegration test、seed固定、doctor拡張、CI向け最小構成。
    - **Non-goals**: 重い学習のフルCI（時間がかかる）。軽量smoke中心。

    ## Depends on
    - depends_on: T071G

    ## Contracts / References
    - docs/00_INVARIANTS.md
- docs/05_ARTIFACT_CONTRACTS.md
- docs/04_CONFIG_SIMPLIFICATION.md
- docs/06_REFACTORING_POLICY.md

    ## Acceptance Criteria
    - [ ] P0〜P2の主要recipeが `pytest -q` でsmoke実行できる（時間上限あり）。
- [ ] 失敗runに対して doctor が原因カテゴリ（IO/contract/sim/graph/ml）を出せる。
- [ ] 乱数seedとhashで再現性が担保される（同一入力で同一出力）。

    ## Verification
    Commands to run (examples):
    - `pytest -q`
- `python -m rxn_platform.cli doctor --strict`

    ## Notes
    - 学習系はミニデータ・少エポックのsmokeに落とす。
