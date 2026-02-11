# GRI30 NetBench（325反応）: ネットワーク構築→縮退→同化の共通テストケース設計

この NetBench は、**GRI-Mech 3.0（gri30.yaml, 325反応）**を用いて、
「ネットワーク構築（Graph）」「縮退（Reduction）」「データ同化（Assimilation）」が
同じテストケース上で評価できるように設計したものです。

---

## 1. なぜ gri30（325反応）が適切か

- **中規模（数百反応）**で、ネットワーク構築・重要度評価・縮退の挙動が十分に表れる
- 燃焼機構は、ラジカル連鎖・循環（SCC）・反応タイプの多様性があり、ネットワーク解析に向く
- Cantera に同梱されるため、**再現性の高いベンチ**にしやすい

---

## 2. 問題設定（0D, 定容/断熱の着火系を想定）

### 状態・入力
- 0D reactor（定容・断熱に近い設定を想定）
- 初期条件を `(T0, P0, phi, t_end)` で与える
- メカニズムは `benchmarks/assets/mechanisms/gri30.yaml`

### 条件セット（複数実行）
- **train**: `benchmarks/assets/conditions/gri30_netbench_train.csv`（24ケース）
  - phi ∈ {0.5, 1.0, 2.0}
  - P ∈ {1, 10} atm
  - T ∈ {1000, 1200, 1400, 1600} K
- **val**: `benchmarks/assets/conditions/gri30_netbench_val.csv`（12ケース）
  - phi ∈ {0.75, 1.5}
  - P ∈ {3, 20} atm
  - T ∈ {1100, 1300, 1500} K

このように、train と val を **別領域（中間値）**にしておくと、
縮退や同化が「学習条件にだけ適合して、一般化しない」問題を検出できます。

---

## 3. ネットワーク構築ベンチ（Graph build）の評価

### 目的
- stoichiometry から **S行列**と **bipartite graph** を作れる
- ノード属性（元素組成・状態）／反応属性（反応タイプ）が付与される
- SCC（循環）が抽出され、循環の強い部分（ラジカル連鎖）を把握できる

### 実行（例）
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_graph \
  run_id=nb_graph_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train
```

---

## 4. 縮退ベンチ（Reduction）の設計：本ケースで十分か？

### 結論
**十分利用できます**。理由は：
- 複数条件で ROP が変化するため、重要度評価の差が出る（削るべき反応が現れる）
- ignition delay / CO2/CO / T_peak を同時に守る必要があり、縮退の難しさが表れる
- val 条件で検証できるので、縮退の過学習（trainのみ維持）を検出できる

### 縮退評価の流れ（推奨）
1) train で重要度推定（ROP積分、感度上位、中心性など）
2) 縮退メカニズム（patch）生成（削除率を複数）
3) **val で再実行**して誤差評価（RMSE/相対誤差/ランキング保存率）

### 他手法比較（例）
- ROP threshold（ベースライン）
- sensitivity top-k preserve
- graph centrality preserve（degree/betweenness）

---

## 5. データ同化ベンチ（Assimilation）の設計：本ケースで十分か？

### 結論
**「全反応係数を同化」するには不十分**ですが、**上位反応（10〜30）に絞れば十分**です。  
（ガス相の少数観測だけで 325反応の全係数を同定するのは不良設定になりがち）

### 同化の推奨設定
- パラメータ：上位K反応の multiplier（K=20〜30）
- 観測（合成でOK）：
  - ignition_delay
  - CO2_final, CO_final
  - T_peak
- train 条件で同化し、val 条件で forward を再評価

### 合成観測生成（推奨）
- `benchmarks/scripts/netbench_generate_obs_cantera.py` を使うと、
  **truth multipliers** を適用した Cantera 直実行で観測CSVを生成できます。

---

## 6. ベンチの実行コマンド（まとめ）

### 0) メカニズム準備
```bash
python3 benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms
```

### 1) Graph build
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_graph \
  run_id=nb_graph_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train
```

### 2) Reduction（複数method比較）
```bash
# まずは sweep（複数 top_k を一括評価）を推奨:
# - runs/<exp>/<run_id>/viz/reduction/ に pass rate / disabled / mean abs diff のSVGが出ます
# - runs/<exp>/<run_id>/viz/network/ に Graphviz の .dot（+可能なら .svg）が出ます
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_rop_sweep \
  run_id=nb_reduce_rop_sweep_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train

python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_centrality_sweep \
  run_id=nb_reduce_centrality_sweep_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train

# ROP threshold
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_rop \
  run_id=nb_reduce_rop_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train

# centrality preserve
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_centrality \
  run_id=nb_reduce_centrality_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train
```

スクリプトでまとめて回す場合は以下:
```bash
BENCH=gri30_netbench_train CASE=tr000 EXP=netbench \
  benchmarks/gri30_netbench/run_reduce_sweep.sh
```

### 3) Assimilation（合成観測→EKI/ES-MDA比較）
```bash
# 合成観測生成（Cantera直）
python3 benchmarks/scripts/netbench_generate_obs_cantera.py \
  --mechanism benchmarks/assets/mechanisms/gri30.yaml \
  --conditions benchmarks/assets/conditions/gri30_netbench_train.csv \
  --truth benchmarks/assets/truth/gri30_truth_multipliers.json \
  --out benchmarks/assets/observations/gri30_netbench_obs.csv

# EKI
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_assim_eki \
  run_id=nb_assim_eki_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train assimilation.obs_file=benchmarks/assets/observations/gri30_netbench_obs.csv

# ES-MDA
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_assim_esmda \
  run_id=nb_assim_esmda_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train assimilation.obs_file=benchmarks/assets/observations/gri30_netbench_obs.csv
```

---

## 7. 実装差異がある場合（重要）

この NetBench が参照する `pipeline` の `task:` 名（例: `sim.sweep_csv`）は、
本基盤の registry key と一致している必要があります。

一致しない場合は:
- `configs/pipeline/bench_gri30_netbench_*.yaml` の `task:` を実装に合わせて置換
- もしくは alias 登録（同じ実装関数を別 key で register）

で合わせてください。

> 注記: `sim.sweep_csv` が未実装の場合は `sim.run_csv` を使用し、`case_id`/`row_index` を
> Hydra の multirun で sweep して複数条件を回す構成に置換してください。
