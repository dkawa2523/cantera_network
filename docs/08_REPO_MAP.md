# 08_REPO_MAP（既存資産マップ）

本ドキュメントは、このリポジトリ内に存在する Cantera 実行資産の
**場所・役割・入出力** を観測ベースで記録する。
観測できない項目は `TODO:` として残す（捏造しない）。

## 1. 走査結果サマリ
- ルート直下には `AGENTS.md`, `docs/`, `work/`, `tools/`, `external/`, `src/`, `configs/`, `artifacts/`, `tests/` があり、隠しディレクトリとして `.git/`, `.venv/`, `.codex/` が **存在し得る**（ローカル環境由来の可能性あり）。
- `rxn_platform` の実装は `src/rxn_platform/` に集約され、ルート直下の `rxn_platform/` は `src/` を参照する shim。
- リポジトリ内で確認できた Python 実行コードは `src/rxn_platform/**.py`、`external/cantera_v1/exampleA.py`、`external/cantera_v1/cantera_semicon_starter_kit/run_cstr_semicon_cantera.py`、`external/cantera_v1/cantera_semicon_starter_kit/make_reactions_from_csv.py`、`external/cantera_v1/cantera_semicon_starter_kit/ck_filter_plasma_to_neutral.py`、`tools/codex_loop/run_loop.py`。
- 確認できた JSON は `work/queue.json`, `work/state.json`, `tools/codex_loop/response_schema.json` で、いずれも運用/スキーマ用途。
- Cantera 既存資産は `external/cantera_v1/` に集約され、スターターキット/修正済み入力セット/例題スクリプト/CSV 出力が存在する。
- 本リポジトリは現時点で **契約/タスク/運用ドキュメント + external の実験素材** が中心。

## 2. 既存のCantera実行入口
- `external/cantera_v1/exampleA.py`: Cantera 付属 `h2o2.yaml` を使った **定圧・断熱バッチ**（0D）。入力はスクリプト内で指定（T0/P0/X0）。出力は stdout と `composition_profiles.csv`、および matplotlib の表示。
- `external/cantera_v1/cantera_semicon_starter_kit/run_cstr_semicon_cantera.py`: **等温 CSTR** の定常近似。入力は `plasma_semicon_neutral.yaml`（ck2yaml で生成想定）と `CASE`/条件パラメータ。出力は stdout のみ（ファイル出力なし）。
- 前処理ユーティリティ（Cantera 実行前）: `external/cantera_v1/cantera_semicon_starter_kit/make_reactions_from_csv.py`（CSV→CHEMKIN `REACTIONS` ブロック）
- 前処理ユーティリティ（Cantera 実行前）: `external/cantera_v1/cantera_semicon_starter_kit/ck_filter_plasma_to_neutral.py`（CHEMKIN から E/e-/PHOTON を除外）
- 参考: `src/rxn_platform/backends/cantera.py` に Cantera backend 実装が存在する。
- TODO: リポジトリ外の既存Cantera資産がある場合は、エントリポイントを追記する。

## 3. 既存出力形式（ログ/CSV/JSON/Artifact）
- `external/cantera_v1/composition_profiles.csv`: `exampleA.py` が出力した CSV。ヘッダは `time_s` + species 名、各行が時刻とモル分率履歴。
- `external/cantera_v1/cantera_semicon_starter_kit/reactions_block.txt`: `make_reactions_from_csv.py` に相当する CHEMKIN 反応ブロックの生成物（テキスト）。
- `run_cstr_semicon_cantera.py` は stdout に定常組成と Arrhenius サンプルを出力（ファイル出力なし）。
- Codex 自動実装ループを回した場合、運用ログが `work/logs/<task_id>_<timestamp>/` に生成され、`prompt.md` や `codex_stdout.txt` / `codex_stderr.txt` / `codex_last_message.json` が確認できる。
- `artifacts/` は存在し、初期状態では `.gitkeep` のみ。RunArtifact などは実行により生成される。
- TODO: リポジトリ外の既存資産がある場合は出力形式を追記する。

## 4. 既存機構ファイルの扱い
- CHEMKIN 入力は `external/cantera_v1/cantera_semicon_starter_kit/` と `external/cantera_v1/cantera_semicon_fix_bundle/` に存在（`chem_neutral*.inp`, `therm*.dat` など）。
- `external/cantera_v1/cantera_semicon_fix_bundle.zip` は修正済み入力セットのアーカイブ。
- 反応リストのサンプルは `external/cantera_v1/cantera_semicon_starter_kit/reactions_sample.csv`。
- Cantera YAML は ck2yaml で生成する想定だが、生成済み YAML はリポジトリ内に無い。
- `external/cantera_v1/exampleA.py` は Cantera 付属の `h2o2.yaml` を利用（リポジトリ内に同梱されていない）。
- TODO: リポジトリ外の既存資産がある場合は、機構ファイル管理方法を追記する。

## 5. 既存の自動実行方式
- Cantera 実行は手動実行（`external/cantera_v1/README.md` と `external/cantera_v1/cantera_semicon_starter_kit/README_ja.md` に手順記載）。CI（`.github/workflows`）や一括スイープは見当たらない。
- `tools/codex_loop/run_loop.py` が `work/queue.json` を読み、Codex 実装ループを回す（ログは `work/logs/` に生成される）。
- TODO: リポジトリ外の既存資産に自動実行（CI/バッチ/スイープ）がある場合は追記する。

## 6. 既存資産と新基盤の共存方針
- `external/cantera_v1/` は legacy 資産として **移動/削除せず保持**する。
- `rxn_platform` への統合は adapter で行い、既存スクリプト/入力セットをラップして Artifact 契約へ写像する方針。
- 既存資産の I/O 形式（CSV/標準出力/生成 YAML）が確定次第、RunArtifact/Observable/Report へのマッピングを追記する。
