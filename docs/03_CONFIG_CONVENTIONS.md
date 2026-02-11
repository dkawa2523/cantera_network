# 03_CONFIG_CONVENTIONS（Hydra設定規約）

本プロジェクトは Hydra を前提とし、YAMLが「設定の真実」です。

## 1. ディレクトリ構造（推奨）

```
configs/
  default.yaml
  defaults.yaml          # 互換エイリアス（legacy）
  recipe/
    smoke.yaml
    sim_sweep.yaml
  sim/                   # legacy: 個別の詳細設定
  task/                  # legacy: カテゴリ単体の設定
  pipeline/              # legacy: step列（taskを並べる）
```

- **新しい入口**は `default.yaml` + `recipe/*.yaml`
- `recipe/*` が `sim/task/pipeline` の組み合わせを選ぶ
- `sim/task/pipeline` は **互換・再利用のために残す**（詳細は `04_CONFIG_SIMPLIFICATION.md`）

> 現時点このリポジトリには `conf/` や `config/` ディレクトリは存在しない。外部既存リポジトリで使われている場合は、衝突を避けるため `configs/` に寄せる。

## 2. 上書き（override）規約
- CLI overrideは Hydra 標準に従う
  - 単発: `python -m <entry> sim=cantera_0d condition.T=700`
  - スイープ: `-m condition.T=700,750,800`
- “同じ条件が2箇所で定義される”状態を禁止（source of truthは1つ）

## 3. run dir / sweep dir
- `hydra.run.dir` は RunStore 配下（`runs/<exp>/<run_id>/hydra`）に寄せる
- 解析成果物は RunStore 内の `artifacts/` へ出す

## 4. Seed
- `seed` は全カテゴリで共通キーにする（例: `common.seed`）
- 乱数を使う処理は必ず seed を参照する

## 5. 設定の固定（manifest化）
- 実行開始時、Hydraの最終設定（merge後）を `manifest.yaml` に保存する
- YAML内の相対パスは `repo_root` 起点で解決する

## 6. “入力編集→実行→出力→可視化” を YAMLで回す
- `sim` は単体でも `rxn sim run sim=...` のように実行可能
- `pipeline` では `steps` により同じRunArtifactを再利用しながら downstream を実行
