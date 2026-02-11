# ベンチマーク一覧（網羅版）

本ファイルは「各機能カテゴリを網羅的に評価」するためのベンチ案を、**実行可能な形（入力・コマンド・評価）**で整理したものです。  
基本は `gri30.yaml`（GRI-Mech 3.0, 325反応）を使い、**中規模（数百反応）**で評価します。

> `gri30.yaml` は Cantera に同梱される代表的機構です（Cantera docs / GitHub を参照）。
> - Cantera input tutorial: https://cantera.org/stable/userguide/input-tutorial.html
> - Cantera GitHub: https://github.com/Cantera/cantera/blob/main/data/gri30.yaml

---

## 共通の前提

### メカニズム配置
- `benchmarks/assets/mechanisms/gri30.yaml` を用意してください。
- `python3 benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms` でコピーできます。

### 条件セット（case）
- `benchmarks.case` に `gri30_small / gri30_medium / gri30_wide` を指定します。
- 実体は `configs/benchmarks/gri30_*.yaml` に定義しています（必要に応じて変更）。

### 出力（RunStore + Artifact）
- すべての結果は RunStore（`runs/<exp>/<run_id>/`）に保存されます。
- Artifact は `runs/<exp>/<run_id>/artifacts/` 配下です。

---

# B0: ベースライン（最小E2E / 失敗検知）

## 目的
- CLI / Hydra / ArtifactStore / dummy backend が問題なく動くことを確認
- 以後のベンチが壊れた時の「切り分け基準」とする

## コマンド
```bash
python3 -m rxn_platform.cli run pipeline=bench_dummy_smoke run_id=bench_dummy_smoke exp=bench sim=dummy
```

## 評価
- `pytest -q` が通る
- `artifacts/runs/` と `artifacts/reports/` が生成される（最小でもOK）

---

# B1: Cantera自動実行（多条件スイープ）ベンチ

## 目的
- `sim` が多条件スイープを回せる
- run_id / manifest が比較可能性を満たしている（設定が真実）

## 入力
- `benchmarks/assets/mechanisms/gri30.yaml`
- `configs/benchmarks/gri30_small.yaml`（条件点）

## コマンド（例）
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_sim_sweep \
  run_id=bench_gri30_sim_sweep exp=bench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  benchmarks.case=gri30_small
```

## 評価指標
- 成功率（失敗条件点の割合）
- 1ケース当たり実行時間、総実行時間
- state出力の健全性（NaNなし、組成和=1 など）

---

# B2: 反応ネットワーク構築・可視化ベンチ

## 目的
- `graphs` が機構から S行列 / bipartite graph / 属性（元素・状態・反応タイプ）を生成できる
- Chemダッシュボードで「主要種・主要反応」が俯瞰できる

## コマンド
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_network \
  run_id=bench_gri30_network exp=bench \
  sim=cantera_0d sim.mechanism=benchmarks/assets/mechanisms/gri30.yaml
```

## 評価指標
- ノード数 / 反応数 / S行列の整合
- SCC（循環）抽出の有無
- 生成された report の閲覧性（人手レビュー）

---

# B3: 特徴量抽出ベンチ（時系列要約 + ROP/生成消滅）

## 目的
- `features` が時系列を、目的に使える特徴量へ変換できる
- 条件間統計（平均・分散・相関）が取れる

## コマンド
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_features \
  run_id=bench_gri30_features exp=bench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  benchmarks.case=gri30_medium
```

## 評価指標
- features の欠損率（NaN/Infがない）
- 上位反応（ROP積分）ランキングの再現性（seed固定時）
- 条件クラスタ別に “支配経路が変わる” ことを説明できるか（Chem視点）

---

# B4: multiplier感度解析ベンチ（重要反応抽出）

## 目的
- `sensitivity` が multiplier 摂動で目的変数への局所感度を出せる
- 上位K反応で次元削減できる（同化/縮退の入口）

## 入力
- 合成の目的変数（例：CO2終濃度 / CO終濃度 / 温度終値）
  - 本基盤側の Observables が未整備なら、`benchmarks/scripts/evaluate.py` 側で代替算出可

## コマンド
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_sensitivity \
  run_id=bench_gri30_sensitivity exp=bench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  benchmarks.case=gri30_small
```

## 評価指標
- 上位K反応の安定性（条件が少し変わっても上位が大きく変わらない/変わる理由が説明できる）
- 同条件での再実行コスト（回数・総時間）

---

# B5: 反応係数の最適化ベンチ（ベースライン）

## 目的
- ベースライン最適化（random / NSGA-II 等）が回る
- 多目的（例：CO2最大化・CO最小化）で Pareto が形成される

## コマンド
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_opt \
  run_id=bench_gri30_opt exp=bench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  benchmarks.case=gri30_small
```

## 評価指標
- best-so-far の改善曲線
- Pareto front の形状（多目的のトレードオフ）
- 乱数seedで再現できること

---

# B6: データ同化ベンチ（合成観測で回復できるか）

## 目的
- EKI / ES-MDA 等の同化が回り、misfit が減少する
- 合成観測に対し、真値（合成生成で使ったパラメータ）へ回復傾向が出る

## 手順
1) 合成観測データを生成（基準runの目的変数にノイズ付与）
2) 同化 pipeline を実行

### 観測生成
```bash
python3 benchmarks/scripts/generate_synthetic_observations.py \
  --mechanism benchmarks/assets/mechanisms/gri30.yaml \
  --case gri30_small \
  --out benchmarks/assets/observations/gri30_obs.csv
```

### 同化
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_assim \
  run_id=bench_gri30_assim exp=bench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  benchmarks.case=gri30_small \
  assimilation.obs_file=benchmarks/assets/observations/gri30_obs.csv
```

## 評価指標
- misfit の減少（前後比）
- 推定パラメータ（multiplier）のRMSE（真値がある場合）
- 収束性（発散/過補正がない）

---

# B7: 縮退ベンチ（prune → 再実行 → 差分評価）

## 目的
- 重要度に基づく縮退（反応削除/集約）が自動で回る
- 縮退後に “目的変数が変わらない” ことを自動で検証できる

## コマンド
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_reduce_validate \
  run_id=bench_gri30_reduce_validate exp=bench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  benchmarks.case=gri30_medium
```

## 評価指標
- 反応数削減率 / 計算時間短縮率
- 目的変数差分（RMSE、最大誤差）
- 主要反応ランキングの保存率（Jaccardなど）

---

# B8: 統合ベンチ（E2E回帰）

## 目的
- sim → graphs → features → sensitivity → reduction → validation → viz の一連が回る
- 結果が `benchmarks/scripts/evaluate.py` で集計できる

## コマンド
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_e2e \
  run_id=bench_gri30_e2e exp=bench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  benchmarks.case=gri30_small
```

---

# 評価集計（共通）

```bash
python3 benchmarks/scripts/evaluate.py --artifacts artifacts --out benchmarks/reports/summary.md
```

- 主要な run / graph / feature / sensitivity / reduction / report の存在確認
- 可能なら、縮退前後の差分を自動集計（run_id指定で）

> 注：本基盤の artifact パス/命名が契約と異なる場合、`evaluate.py` の探索パターンを調整してください。
