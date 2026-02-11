# Cantera 半導体プロセス機構メモ（スターターキット + 例題）

本リポジトリは、半導体プロセス向けのガス相反応機構を **CHEMKIN -> Cantera(YAML)** に載せて
等温 CSTR の定常組成を確認するためのスターターキットと、
Cantera の基本動作を確認するための H2/O2 例題をまとめたものです。

主に以下の 2 系統が入っています。
- **cantera_semicon_starter_kit/**: CF4/O2(/N2), CHF3/C2F6/C4F8 系の中性反応機構を作るためのテンプレートとスクリプト
- **exampleA.py**: Cantera の基本動作確認用（H2/O2 定圧・断熱バッチ）

---

## 全体ワークフロー（スターターキット）
1. 反応式と Arrhenius 係数を CSV 化（`reactions_sample.csv` の形式）。
2. `make_reactions_from_csv.py` で CHEMKIN の `REACTIONS` ブロックを生成。
3. `chem_stub.inp` を埋めて `chem.inp`（または `chem_neutral.inp`）を作成。
4. 必要なら `ck_filter_plasma_to_neutral.py` で E/e-/PHOTON を除外。
5. `ck2yaml` で YAML 化し、`run_cstr_semicon_cantera.py` で等温 CSTR を実行。

注意:
- `make_reactions_from_csv.py` は `REACTIONS KCAL/MOLE ...` のヘッダを出します。
  CSV の `Ea_over_R` を **K のまま**入れると単位不整合になります。
  変換済みの例は `chem_neutral_fixed.inp` や `reactions_block_kcal.txt` を参照してください。
- `therm.dat` / `therm_valid.dat` は **プレースホルダ**です。実解析には一次資料から NASA 係数を準備する必要があります。

---

## ルート直下の構成
- `cantera_semicon_starter_kit/`: 半導体プロセス機構のスターターキット本体
- `cantera_semicon_fix_bundle/`: 単位変換や ck2yaml 通過用に整形した「修正済み」入力セット
- `cantera_semicon_fix_bundle.zip`: 上記 fix_bundle のアーカイブ
- `exampleA.py`: H2/O2 定圧・断熱バッチの Cantera サンプル（グラフ出力 + CSV 保存）
- `composition_profiles.csv`: `exampleA.py` が生成したモル分率履歴（時間列）
- `.venv/`: ローカル Python 仮想環境（プロジェクト本体ではない）

---

## `cantera_semicon_starter_kit/` の中身
- `.gitignore`: venv や生成物（YAML, reactions_block, chem_neutral など）を除外
- `README_ja.md`: スターターキットの詳細手順（一次資料からの整形、ck2yaml 実行など）
- `ck_filter_plasma_to_neutral.py`: CHEMKIN 入力から `E/e-/PHOTON` を含む種・反応を除外
- `make_reactions_from_csv.py`: CSV から `REACTIONS` ブロックを生成（Arrhenius A, b, Ea）
- `run_cstr_semicon_cantera.py`: 等温 CSTR を実行し、定常組成・反応数を出力
- `chem_stub.inp`: CHEMKIN 入力の骨格（反応はダミー 1 本のみ）
- `chem_neutral.inp`: 中性反応のみの例（Ea が K のままのサンプル）
- `chem_neutral_fixed.inp`: `chem_neutral.inp` の Ea を kcal/mol に変換した版
- `reactions_sample.csv`: CSV 形式のサンプル反応リスト
- `reactions_block.txt`: `reactions_sample.csv` から生成した `REACTIONS` ブロック
- `therm_stub.dat`: NASA 7 係数の骨格テンプレート
- `therm.dat`: プレースホルダ NASA 係数（実解析用ではない）
- `therm_valid.dat`: 要素組成を含む形式に合わせたプレースホルダ（ck2yaml 通過用）
- `requirements.txt`: Python 依存（cantera, matplotlib, numpy）

---

## `cantera_semicon_fix_bundle/` の中身
- `chem_neutral.inp`: 中性反応のみの CHEMKIN 入力（ベース）
- `chem_neutral_fixed.inp`: Ea を kcal/mol に変換済みの CHEMKIN 入力
- `chem_neutral_ck2yaml_ready.inp`: THERMO ブロック無しで ck2yaml 直実行できる形に整形
- `reactions_block.txt`: 反応ブロック（K のまま）
- `reactions_block_kcal.txt`: 反応ブロック（Ea を kcal/mol に変換済み）
- `therm.dat`: プレースホルダの NASA 係数
- `therm_valid.dat`: 要素組成つきの NASA 係数フォーマット（ck2yaml 用）

---

## `exampleA.py` / `composition_profiles.csv`
- `exampleA.py` は、Cantera 同梱の `h2o2.yaml` を使った
  **定圧・断熱バッチ反応**の最小例です。
- 実行すると、温度と主要種の時間変化を描画し、モル分率履歴を
  `composition_profiles.csv` に保存します。
