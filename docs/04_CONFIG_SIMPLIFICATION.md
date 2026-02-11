# 04_CONFIG_SIMPLIFICATION（設定/出力の簡素化）

このドキュメントは「YAML爆発」と「出力階層の散逸」を抑えるためのルールです。
新しい入口は **default.yaml + recipe/** であり、出力は **RunStore** に集約します。

## 1. 新しい入口（default + recipe）
- **基本コマンド**
  ```bash
  python -m rxn_platform.cli run recipe=smoke run_id=demo_001
  ```
- 実行履歴の確認
  ```bash
  python -m rxn_platform.cli list-runs --last 5
  ```
- `default.yaml` は最小の共通設定（seed / run / store / hydra）だけを持つ
- `recipe/*.yaml` が具体的な `sim/task/pipeline` の組み合わせを選ぶ

> 既存の `sim/` `task/` `pipeline/` は **互換・再利用のために残す**。
> ただし新規追加は recipe を入口にする。

## 2. RunStore（出力の唯一の入口）
RunStore は `runs/<exp>/<run_id>/` に集約する。

```
runs/
  <exp>/
    <run_id>/
      manifest.json
      config_resolved.yaml
      metrics.json
      artifacts/
        runs/<artifact_id>/
        observables/<artifact_id>/
        ...
      hydra/
```

- `manifest.json`: RunStore のメタ情報（recipe / created_at / store_root）
- `config_resolved.yaml`: 実行時の最終設定（Hydra merge 後）
- `metrics.json`: pipeline/task の結果（artifact_id 参照）
- `artifacts/` 以下は **02_ARTIFACT_CONTRACTS に準拠**
- `cli run` は `store.root` を RunStore 配下に強制する（散逸防止）
- RunStore の `run_id` は **ラベル**であり、Artifact の id は hash-based のまま

### 2.1 必須のRunStore成果物
- `manifest.json`
- `config_resolved.yaml`
- `metrics.json`
- `sim/timeseries.zarr`（RunArtifact）
- `graphs/meta.json`（グラフ生成済みの場合）
- `viz/`（レポート実行済みの場合）

### 2.2 条件テーブル（cases）の同期
- `conditions_path` / `conditions_csv` が設定されている場合は、`runs/<exp>/<run_id>/inputs/conditions.(csv|parquet)` に同期する。
- 同期したパスは `manifest.json` の `conditions_path` に記録する。

## 3. 旧構成からの移行
### 3.1 config 名
- 旧: `defaults.yaml`
- 新: `default.yaml`（`defaults.yaml` は互換エイリアス）

### 3.2 旧 Hydra group から recipe へ
- 旧: `sim=... pipeline=... task=...`
- 新: `recipe=...` で組み合わせを指定

互換のため `sim/task/pipeline` group は残すが、**新規は recipe に集約**する。

### 3.4 Legacy yaml 移行ガイド（概要）
1) 既存 `sim/task/pipeline` 設定を確認し、最短の `recipe` を作る。
2) `recipe` から `sim/task/pipeline` を参照する形にして、入口を固定する。
3) CI/運用コマンドは `run recipe=...` に統一する。
4) 互換運用のまま残る legacy yaml は一覧化し、新規追加を禁止する。
5) CLIは `sim/task/pipeline` の直接指定を警告する（互換は維持）。

### 3.3 出力の移行
- 旧: `artifacts/` 直下に散在
- 新: `runs/<exp>/<run_id>/artifacts/` に集約

既存の成果物は以下のいずれかで移行:
- `artifacts/` を `runs/<exp>/<run_id>/artifacts/` へコピー
- 既存運用のまま使う場合は `store.root=artifacts` を明示（互換運用）

## 4. 互換運用ポリシー
- `recipe` は **短い YAML** に限定する
- 深い group ネストは避ける（残す場合は理由を明記）
- `sim/task/pipeline` は **後方互換のために維持**（将来削除予定）

### 4.1 Legacy YAML 一覧（新規追加禁止）
- `configs/sim/*`
- `configs/task/*`
- `configs/pipeline/*`
- `configs/benchmarks*/*`
新規の設定追加は `configs/recipe/*` にのみ行う。

## 5. 実装 TODO
- `sim_sweep` recipe は現状 smoke pipeline のプレースホルダ
- `sim.sweep_csv` 実装後に実スイープへ置換する
