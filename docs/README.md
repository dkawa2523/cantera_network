# docs/ README（読む順）

この docs/ は **開発中に変えない契約（Invariants / Contracts）** を集約します。
Codex / 開発者の全実装は、この docs/ に従います。

> 注意: 現在の既存リポジトリ（例: cantera自動実行テストコード）をここでは実際に読み取れていません。
> リポジトリ固有の事情は **捏造せず** `TODO:` として記載しています。

## 読む順（最短）
1. `00_INVARIANTS.md`
2. `01_ARCHITECTURE.md`
3. `02_ARTIFACT_CONTRACTS.md`
4. `03_CONFIG_CONVENTIONS.md`
5. `04_PIPELINES.md`
6. `05_VISUALIZATION_STANDARDS.md`
7. `06_TESTING_AND_QA.md`
8. `07_CONTRIBUTING.md`

## この基盤が目指すこと（要約）
- **シミュレーションは差し替え可能なBackend**（Cantera以外も将来置換可能）
- **解析カテゴリ（グラフ/特徴量/感度/ML/最適化/縮退/可視化）は独立実行**
- **カテゴリ間の接続は Artifact（成果物）契約のみ**（スパゲッティ防止）
- Hydra YAML が「設定の真実」。実行は YAML から再現可能。
- 多目的観測（膜厚、気体組成、占有率等）を **後から追加・変更しやすい**
- 多条件自動実行を活かし、同化・縮退・最適化を **自動回帰テスト**で安全に回す

## まず実装で固定する要点（MVP）
- `Artifact` / `Store` / `Task` の共通I/O
- `SimulationBackend` インターフェース（Canteraは実装の1つ）
- `Observable` プラグイン（膜厚/組成/占有率…）
- `Pipeline Runner`（YAML steps の順次実行）
- `Golden condition suite` による回帰テスト

## TODO（リポジトリ固有）
- 既存リポジトリのディレクトリ構成・エントリポイントの現状把握
- 既存Cantera実行コードの入出力・ログ形式の把握
- 既存の機構ファイル管理方法（yaml/cti/chemkin）
- 既存の可視化（ある場合）の統合方針
