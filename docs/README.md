# docs/ README（読む順）

この docs/ は **開発中に変えない契約（Invariants / Contracts）** を集約します。
Codex / 開発者の全実装は、この docs/ に従います。

> 注意: `external/cantera_v1/` に Cantera の既存サンプル/スターターキット/CSV 出力が存在します（詳細は `08_REPO_MAP.md`）。
> `rxn_platform` 本体の実装は `src/rxn_platform/` にあります（ルート直下の `rxn_platform/` は shim）。

## 読む順（最短）
1. `00_INVARIANTS.md`
2. `01_ARCHITECTURE.md`
3. `02_ARTIFACT_CONTRACTS.md`
4. `03_CONFIG_CONVENTIONS.md`
5. `04_PIPELINES.md`
6. `05_VISUALIZATION_STANDARDS.md`
7. `06_TESTING_AND_QA.md`
8. `07_CONTRIBUTING.md`
9. `08_REPO_MAP.md`

## クイックスタート（smoke まで）
前提: Python 3.9+ と依存関係（`hydra-core` など）が利用可能なこと。
ネットワークに依存しない環境の場合は、社内ミラーやローカルwheelで依存を用意すること（TODO: 手順追記）。

1) doctor を実行（環境診断 + dummy smoke）
```bash
python -m rxn_platform.cli doctor
```

2) smoke pipeline を実行（dummy backend）
```bash
python -m rxn_platform.cli pipeline run pipeline=smoke sim=dummy
```
出力 JSON の `results.sim` が `run_id`。

3) artifacts を一覧表示
```bash
python -m rxn_platform.cli artifacts ls --root artifacts
```

4) run を可視化（report を生成）
```bash
python -m rxn_platform.cli sim viz <run_id>
python -m rxn_platform.cli artifacts open-report <report_id>
```

## 簡易検証
```bash
pytest -q
```

## 拡張時のルール（タスク追加 / 契約更新）
- 新規タスクは `work/queue.json` と `work/tasks/Txxx_*.md` をセットで追加する（Acceptance Criteria と Verification を必須で書く）。
- docs の契約変更は原則禁止。どうしても必要な場合は「互換性維持」「移行手順」「影響範囲」を必ず記す（詳細は `07_CONTRIBUTING.md`）。
- 追加ドキュメントが契約カテゴリに該当する場合は、この README の読む順に追記する。

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
- リポジトリ直下には `AGENTS.md`, `docs/`, `work/`, `tools/`, `external/`, `src/`, `configs/`, `artifacts/`, `tests/` があり、隠しディレクトリとして `.codex/`, `.git/`, `.venv/` が存在する（ローカル環境由来の可能性あり）。詳細は `08_REPO_MAP.md`。TODO: リポジトリ外に追加の既存資産があれば追記。
- 既存Cantera実行コードは `external/cantera_v1/exampleA.py` と `external/cantera_v1/cantera_semicon_starter_kit/run_cstr_semicon_cantera.py` に存在し、出力は `external/cantera_v1/composition_profiles.csv`（時系列 CSV）や stdout で確認できる。TODO: rxn_platform への統合方法を追記。
- 既存の機構ファイルは `external/cantera_v1/cantera_semicon_starter_kit/` と `external/cantera_v1/cantera_semicon_fix_bundle/` の CHEMKIN `.inp` / `therm*.dat` にある（YAML は ck2yaml で生成予定）。TODO: 生成 YAML の保存場所と命名を追記。
- 可視化は `external/cantera_v1/exampleA.py` が matplotlib で描画するが、保存された図ファイルは見当たらない。TODO: 図の保存先やレポート化の方針を追記。
- TODO: SBI を使う場合は `sbi`（および依存の `torch`）を別途用意し、`rxn_platform.tasks.sbi` の SNPE 実行手順を追記する。
