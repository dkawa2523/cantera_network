# Cantera 半導体プロセス機構スターターキット（CF₄/O₂(/N₂)・CHF₃/C₂F₆/C₄F₈）

このキットは、**CHEMKIN 形式の機構（反応100本以上）**を Cantera で使うための
最小構成テンプレートとユーティリティスクリプトを含みます。

> 目的：手元に `chem.inp` / `therm.dat` が無い状態から、
> 公開一次情報（例：SAND2001-1292、Levko 2021）をもとにファイルを整形 →
> `ck2yaml` で YAML 化 → **等温CSTR** で**反応組成（定常近似）**を求める。

---

## 構成

```
ck_filter_plasma_to_neutral.py   # E/e-/PHOTON を含む反応・種を CHEMKIN から除外
run_cstr_semicon_cantera.py      # Cantera 等温CSTR 実行（YAML 読み込み）
make_reactions_from_csv.py       # CSV から CHEMKIN REACTIONS ブロック生成
chem_stub.inp                    # CHEMKIN 機構の骨格（反応はダミー1行のみ）
therm_stub.dat                   # NASA 係数の骨格（中身は未記入）
reactions_sample.csv             # ダミーの中性反応サンプル（10反応）
requirements.txt
```

> **注意**：`chem_stub.inp` と `therm_stub.dat` は**骨格**です。
> 実際に回すには、公開一次資料から **反応（100+）** と **NASA係数** を追加してください。
> まずは **中性反応のみ**で YAML を作るのが安定です。

---

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Cantera の `ck2yaml` が使える状態になります。

---

## 手順（最短）

### 1) 反応リストの準備（中性反応のみ）
- 例：SAND2001-1292 の Table 2/9/13、Levko 2021 の Appendix から**中性反応**を CSV に整形します。
- サンプル形式：`reactions_sample.csv` を参照（ヘッダ：`equation,A,b,Ea_over_R,units`）。
  - `equation`: `CF2 + O = CFO + F` など
  - `A`: 前因子
  - `b`: 温度指数
  - `Ea_over_R`: 活性化エネルギーを気体定数 R で割った値 [K]
  - `units`: 空欄推奨（必要ならメモ用途で可）

CSV から CHEMKIN 行を生成：
```bash
python make_reactions_from_csv.py reactions_sample.csv > reactions_block.txt
```

### 2) `chem.inp` の作成
- `chem_stub.inp` をベースに、`REACTIONS ... END` の間を `reactions_block.txt` で置換します。
- **電子・光子反応が混ざっている**場合は、後述のフィルタで除外してください。

### 3) `therm.dat` の作成
- `therm_stub.dat` に、必要種（CF4, CF3, CF2, CFO, COF2, CO, CO2, O2, O, F, N2, …）の NASA 7 係数を追記します。
- 既存公開の NASA 多項式（Burcat/ATcT 由来など）から転記するのが一般的です。

### 4) **（任意）** E/e-/PHOTON を含む CHEMKIN から**中性のみ**を抽出
```bash
python ck_filter_plasma_to_neutral.py chem.inp chem_neutral.inp
```

### 5) YAML 化（ck2yaml）
```bash
# 中性のみでまず作ることを推奨
ck2yaml --input chem_neutral.inp --thermo therm.dat --output plasma_semicon_neutral.yaml

# そのまま行く場合
ck2yaml --input chem.inp --thermo therm.dat --output plasma_semicon.yaml
```

### 6) Cantera で等温CSTRを実行
```bash
# YAML ファイル名と流入組成 CASE を調整してから
python run_cstr_semicon_cantera.py
```

出力の冒頭に `#species` と `#reactions` が表示されます（**100+反応**が読めていることを確認）。
最後に CSTR の**定常近似のモル分率**が出力されます。

---

## よくあるポイント

- **まず中性反応で土台**：E/e-/PHOTON を含むと、Cantera/ck2yaml の取り扱いが難しくなります。
- **圧力・温度**：mTorr オーダ＋等温でベースライン → 必要に応じて `energy="on"` や表面反応を追加。
- **種名の表記差**：機構ごとに `C4F8` vs `NC4F8` のような違いがあります。`CASE` の組成で機構の種名に合わせてください。

---

## 免責
- `reactions_sample.csv` は**ダミー**です（生化学的妥当性は保証しません）。
- 実解析には**一次資料**の反応セット（100+）と NASA 係数を用いてください。
