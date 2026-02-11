# Benchmarks（反応ネットワーク解析基盤の評価スイート）

このディレクトリは、構築済みの Cantera 自動実行＋反応ネットワーク解析基盤（以下「本基盤」）を
**ある程度の粒度で評価・ベンチマーク**するための「例題・設定・実行手順・評価スクリプト」をまとめたものです。

## 目的
- `sim / graphs / features / sensitivity / assimilation / optimization / reduction / viz` の各カテゴリが
  仕様通りに動き、結果が比較可能であることを確認する
- 目的変数（膜厚 / 組成 / 占有率 / 任意QoI）を後から追加・変更しても、ベンチが壊れにくい構成にする
- 中規模〜大規模（**数十〜数百反応**）の機構で、計算が回り、重要度・縮退・同化・最適化が評価できることを確認する

## まず最初に（必須）
1. メカニズムを用意する（推奨：Cantera同梱 `gri30.yaml`）
2. `benchmarks/scripts/setup_mechanisms.py` を実行して `benchmarks/assets/mechanisms/` にコピーする

```bash
python3 benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms
```

> Cantera の data ディレクトリの場所はインストール環境により異なります。
> Cantera 公式ドキュメントでは、dataファイルはインストール先の `data/` 配下にある旨が説明されています。
> また `gri30.yaml` は Cantera に同梱される代表的機構です。出典: Cantera docs, Cantera GitHub。  

> - https://cantera.org/stable/userguide/input-tutorial.html  

> - https://github.com/Cantera/cantera/blob/main/data/gri30.yaml

## 実行方法（典型）
本基盤の CLI が `python -m rxn_platform.cli ...` 形式の場合：

```bash
# 例：ネットワーク構築ベンチ
python3 -m rxn_platform.cli run pipeline=bench_gri30_network \
  run_id=bench_gri30_network exp=bench \
  sim=cantera_0d sim.mechanism=benchmarks/assets/mechanisms/gri30.yaml
```

CLI が `rxn ...` 形式の場合は、上記の `python -m rxn_platform.cli` を `rxn` に置き換えてください。

## 何が入っているか
- `benchmarks/BENCHMARKS.md`：ベンチの一覧と目的・評価指標・実行コマンド
- `benchmarks/assets/conditions/`：条件スイープ例（CSV/YAML）
- `benchmarks/assets/observations/`：同化用の観測データ（※合成生成の雛形）
- `benchmarks/scripts/`：メカニズム準備、観測生成、評価集計のスクリプト
- `configs/pipeline/bench_*.yaml`：Hydra の pipeline 定義（本基盤の命名に合わせて調整してください）
- `configs/benchmarks/*.yaml`：条件セット（case）定義

## 重要：このベンチは「契約」に従う
- 本基盤の `docs/00_INVARIANTS.md` と `docs/02_ARTIFACT_CONTRACTS.md`（Artifact契約）に従います。
- ベンチの出力先は RunStore（`runs/<exp>/<run_id>/`）であり、Artifact は `runs/<exp>/<run_id>/artifacts/` 配下に保存されます。
