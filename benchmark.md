# Benchmark 可視化ガイド

このリポジトリのベンチマーク比較・評価を「人が理解しやすい形」にするための可視化をまとめます。
以下の可視化は `viz.benchmark_report` と `viz.chem_dashboard` に追加済みです。

## 追加した可視化（最低 5 つ以上）

### 1. Benchmark Comparisons（比較チャート）
- **Optimization Objective Comparison**: 最終目的関数値の比較（複数 objective の場合は系列で比較）。
- **Assimilation Final Misfit Comparison**: 同化の最終ミスフィット（mean/min/max）の比較。
- **Validation Pass Rate Comparison**: 検証 pass rate の比較。
- **Reduction Size vs Pass Rate**: 縮退サイズ（無効化反応数）と pass rate の相関。

### 2. Mechanism Networks (Graphviz)
- **Top ROP Reaction Network**: 重要反応（ROP 上位）を中心にしたネットワーク。
- **Top WDOT Species Network**: 重要種（WDOT 上位）を中心にしたネットワーク。
- **Reduction Patch Network**: 縮退で無効化された反応がどの種・反応に繋がるか。
- **Assimilation Parameter Network**: 同化で調整される反応パラメータがどの反応に対応するか。

> Graphviz は環境によって `dot` が無い場合があります。
> その場合は **DOT ソースを埋め込み表示**し、後で `dot -Tsvg` などで生成できます。

## 実行例

### ベンチマーク比較レポート
```
python3 -m rxn_platform.cli run pipeline=p2_smoke run_id=p2_smoke exp=bench sim=dummy
```
または `viz.benchmark_report` を直接使う場合は、比較したい Artifact を inputs に渡します。

### Chem Dashboard（Cantera 実行結果の理解）
```
python3 -m rxn_platform.cli run pipeline=bench_gri30_network \
  run_id=bench_gri30_network exp=bench \
  sim=cantera_0d sim.mechanism=benchmarks/assets/mechanisms/gri30.yaml
```

## グラフ可読性のための設定（任意）
`viz.graphviz` に以下の設定を追加できます（任意）。

```
# 例: configs で指定
viz:
  graphviz:
    engine: dot
    top_n: 8
    max_nodes: 80
    max_edges: 160
    rop_stat: integral
    wdot_stat: integral
```

- **engine**: `dot` / `sfdp` など。`dot` を推奨。
- **top_n**: 上位反応・種の選択数。
- **max_nodes / max_edges**: 失敗しやすいレイアウトを避けるための上限。


# ベンチマーク本番


1) このテストケースの狙い（問題設定）

ユーザー要望に沿って、「ネットワーク構築ベンチ」単体ではなく、同じ問題設定を使って

ネットワーク構築（Graph build）

縮退（Reduction）：複数回自動実行→縮退→再実行→差分評価

データ同化（Assimilation）：複数条件の合成観測→推定→汎化検証

他手法比較：縮退・同化を複数手法で並列比較

まで、ひとつの共通ベンチとして回せるようにしています。

2) 「gri30（325反応）」で縮退・同化が十分に評価できるか？
結論

縮退：十分に評価可能
325反応でも、条件（T/P/phi）を振ると支配反応・経路が変わり、
ROP/感度/中心性で“削って良い反応”が出るため、縮退→再実行→誤差評価の差がちゃんと出ます。

**同化：全反応係数の推定は不適（観測が少なすぎる）**だが、
「上位反応（10〜30）」に絞る同化なら十分に評価できます。
→ NetBenchでは「感度で top-k を抽出 → EKI/ES-MDAで multiplier 推定」構成にしています。

3) 追加パックが入れているファイル（実行に必要なもの）

展開すると、以下がリポジトリに “上乗せ” されます。

条件ファイル（複数実行の設計）

benchmarks/assets/conditions/gri30_netbench_train.csv（24ケース）

phi ∈ {0.5, 1.0, 2.0} × P ∈ {1,10 atm} × T ∈ {1000,1200,1400,1600 K}

benchmarks/assets/conditions/gri30_netbench_val.csv（12ケース）

phi ∈ {0.75, 1.5} × P ∈ {3,20 atm} × T ∈ {1100,1300,1500 K}

train と val が “中間値” の別領域になっているため、
縮退・同化の過学習（trainだけ合う）が検出できます。

ベンチ定義（Hydra）

configs/benchmarks/gri30_netbench_train.yaml

configs/benchmarks/gri30_netbench_val.yaml

pipeline（比較可能な形）

configs/pipeline/bench_gri30_netbench_graph.yaml
→ ネットワーク構築・可視化

configs/pipeline/bench_gri30_netbench_reduce_rop.yaml
→ 縮退（ROP threshold ベースライン）＋ val検証

configs/pipeline/bench_gri30_netbench_reduce_centrality.yaml
→ 縮退（中心性 preserve）＋ val検証（別手法比較）

configs/pipeline/bench_gri30_netbench_assim_eki.yaml
→ 同化（EKI）

configs/pipeline/bench_gri30_netbench_assim_esmda.yaml
→ 同化（ES-MDA）で比較

合成観測（同化用）

benchmarks/assets/truth/gri30_truth_multipliers.json
→ “真値”の multiplier（反応式ベースで指定）

benchmarks/scripts/netbench_generate_obs_cantera.py
→ Cantera直で合成観測CSVを生成（プラットフォーム未実装でも観測生成できる）

benchmarks/assets/observations/gri30_netbench_obs_README.md

実行まとめ

benchmarks/gri30_netbench/NETBENCH.md（問題設定・評価・手順まとめ）

benchmarks/gri30_netbench/run_all.sh（比較を順に流す補助）

4) 実行手順（最短）
4.1 パック適用

あなたのリポジトリルートで：

cd <repo_root>
unzip -o rxn_gri30_netbench_case_pack.zip -d .

4.2 gri30.yaml を用意

（既に前のベンチパックがある前提なら同じ手順でOK）

python3 benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms

4.3 ネットワーク構築ベンチ（train）
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_graph \
  run_id=nb_graph_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train

4.4 縮退ベンチ（手法比較：ROP / centrality）
# ROP threshold（ベースライン）
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_rop \
  run_id=nb_reduce_rop_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train

# centrality preserve（比較手法）
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_reduce_centrality \
  run_id=nb_reduce_centrality_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train

4.5 同化ベンチ（合成観測生成 → EKI/ES-MDA比較）
合成観測生成（Cantera直）
python3 benchmarks/scripts/netbench_generate_obs_cantera.py \
  --mechanism benchmarks/assets/mechanisms/gri30.yaml \
  --conditions benchmarks/assets/conditions/gri30_netbench_train.csv \
  --truth benchmarks/assets/truth/gri30_truth_multipliers.json \
  --out benchmarks/assets/observations/gri30_netbench_obs.csv

同化（EKI）
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_assim_eki \
  run_id=nb_assim_eki_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train \
  assimilation.obs_file=benchmarks/assets/observations/gri30_netbench_obs.csv

同化（ES-MDA）
python3 -m rxn_platform.cli run pipeline=bench_gri30_netbench_assim_esmda \
  run_id=nb_assim_esmda_tr000 exp=netbench \
  sim=cantera_0d mechanism.path=benchmarks/assets/mechanisms/gri30.yaml \
  +benchmarks=gri30_netbench_train \
  assimilation.obs_file=benchmarks/assets/observations/gri30_netbench_obs.csv

5) どこが「比較検討できる形」になっているか
縮退比較

同じ train/val、同じ評価指標（ignition_delay / T_peak / CO2_final / CO_final）で
**手法だけ変えて（ROP vs centrality）**性能差が出ます。

同化比較

同じ観測CSVに対して
EKI / ES-MDA の収束性、misfit低下、パラメータ推定の安定性を比較できます。

さらにベースライン比較を追加したい場合は：

同化を使わず random / CMA-ES / Bayesian optimization で multiplier をフィット
を同じ観測で回すことで比較できます（今後追加しやすい設計）

6) 重要：あなたの実装の task 名が違う場合

このパックの configs/pipeline/bench_gri30_netbench_*.yaml にある task: 名は、
以前の設計案の想定（例 sim.sweep_csv, graphs.build など）です。

もしあなたの実装で registry key が違う場合は、

configs/pipeline/bench_gri30_netbench_*.yaml の task: を実装名に合わせて置換

してください（このパックは “問題設定” と “比較可能な条件設計” が本体なので、
タスク名は置換で合わせる運用が現実的です）。

必要なら次のステップとして、あなたの実装済みリポジトリの 実際の task registry 名（一覧）に合わせて、
この netbench 用 pipeline yaml を 完全に一致する名前に自動変換した版の zip を作り直します。

> 注記: `sim.sweep_csv` が未実装の場合は `sim.run_csv` を使い、`case_id`/`row_index` を
> Hydra の multirun で sweep する構成に置換してください。
