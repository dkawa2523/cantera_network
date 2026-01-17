# EPICS / 開発ロードマップ（タスク地図）

このファイルは **work/queue.json** を読むための「人間向けの索引」です。
実際の実行順序は `depends_on` と `priority` に従います（source of truth は queue.json）。

## 目的と対応カテゴリ

- **シミュレーション自動実行**（backend差し替え可能）
  - P0: T011（DummyBackend）, T024（sim CLI）
  - P1: T021–T023（CanteraBackend + 出力 + multipliers）
- **反応ネットワーク化 / 特徴量化**
  - Graph: T029–T033
  - Feature: T034–T036
- **感度解析（重要反応抽出）**
  - T037–T038
- **最適化 / データ同化**
  - Optimization: T041–T042
  - Assimilation: T043–T045
- **縮退（削除/集約）と妥当性評価**
  - Patch: T046
  - Prune/Validate: T047–T048
  - Lumping候補: T049–T050
- **可視化（DS視点 / 化学視点）**
  - Base: T018
  - DS dashboard: T039
  - Chem dashboard: T040
- **QA/回帰**
  - Smoke/Doctor: T016–T017
  - Golden suite: T051
  - Benchmark: T052
- **フェーズゲート（P0/P1/P2/P3間の統合テストと修正）**
  - P0→P1: T020G
  - P1→P2: T040G
  - P2→P3: T052G
  - P3→Release: T060G
- **発展（Laplacian, MBDoE, Active Subspace, SBI, GNN dataset, Multi-fidelity）**
  - T053–T059

## 推奨の最短ルート（まず動かす）

1. P0基盤: T001 → T002 → T003 → T004 → T005 → T007 → T009 → T010 → T011 → T012 → T013 → T014 → T015 → T016 → T017 → **T020G**
2. P1主要機能: T021 → T022 → T023 → T024 → T025 → T026 → T028 → T029 → T030 → T031 → T032 → T033 → T034 → T035 → T037 → T039/T040 → **T040G**
3. P2中核: T046 → T047 → T048 → T041 → T043 → T044 → T051 → T052 → **T052G**
4. P3発展（任意）: T053 → T054 → T055 → T058 → T059 → T060 → **T060G**

## データサイエンティスト視点の“最低限ほしい出力”

- 条件メタデータ（run manifestから）
- 目的変数ベクトル（ObservableArtifact）
- 感度ランキング（SensitivityArtifact）
- 最適化/同化の履歴（OptimizationArtifact / PosteriorArtifact）
- before/after差分（ValidationArtifact）
- それらのHTMLレポート（ReportArtifact）

## 化学反応専門家視点の“最低限ほしい出力”

- 反応式とIDの対応表、reaction_type、可逆性
- 主要種/主要反応の time series, ROPランキング
- SCC/循環構造・重要なループ
- 縮退で“何を消したか”の説明（patch）と影響（validation）
