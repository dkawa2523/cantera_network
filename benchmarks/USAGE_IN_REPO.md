# 既存リポジトリでの配置・利用方法

## 配置（推奨）
この zip は「リポジトリルートに展開」して使う想定です。

```
<repo_root>/
  benchmarks/                 # ← 本zipで追加
  configs/benchmarks/         # ← 本zipで追加（Hydra group）
  configs/pipeline/bench_*.yaml
  runs/                       # ← RunStore 出力（契約）
```

既存リポジトリに `configs/` がある場合は上書きではなくマージしてください。

## 使い方（最短）
1) メカニズム準備
```bash
python3 benchmarks/scripts/setup_mechanisms.py --dest benchmarks/assets/mechanisms
```

2) Dummy smoke（環境が整っているか）
```bash
python3 -m rxn_platform.cli run pipeline=bench_dummy_smoke run_id=bench_dummy_smoke exp=bench sim=dummy
```

3) GRI3.0でネットワーク構築
```bash
python3 -m rxn_platform.cli run pipeline=bench_gri30_network \
  run_id=bench_gri30_network exp=bench \
  sim=cantera_0d sim.mechanism=benchmarks/assets/mechanisms/gri30.yaml
```

4) 結果集計
```bash
python3 benchmarks/scripts/evaluate.py --artifacts artifacts --out benchmarks/reports/summary.md
```

## 注意（本基盤の実装差異がある場合）
- pipeline の task 名（例: `sim.sweep_csv`）は本zipの想定です。
  実際の registry key と異なる場合は、`configs/pipeline/bench_*.yaml` の `task:` を調整してください。
- 同様に、sim の条件入力（CSVカラム名）も実装に合わせて読み替えが必要な場合があります。
  その場合は `benchmarks/assets/conditions/*.csv` を実装仕様に合わせて更新してください。

## 期待される成果物
- `runs/<exp>/<run_id>/manifest.json` / `config_resolved.yaml` / `metrics.json` / `summary.json`
- `runs/<exp>/<run_id>/artifacts/<kind>/<artifact_id>/...`
- `runs/<exp>/<run_id>/viz/index.html`
- `runs/<exp>/<run_id>/viz/network/index.json` + `*.dot`（Graphviz があれば `*.svg`）
