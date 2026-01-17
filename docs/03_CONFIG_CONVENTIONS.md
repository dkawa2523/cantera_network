# 03_CONFIG_CONVENTIONS（Hydra設定規約）

本プロジェクトは Hydra を前提とし、YAMLが「設定の真実」です。

## 1. ディレクトリ構造（推奨）

```
configs/
  defaults.yaml
  sim/
    cantera_0d.yaml
    dummy.yaml
  task/
    graphs.yaml
    features.yaml
    sensitivity.yaml
    assimilation_eki.yaml
    reduction_threshold.yaml
    viz_ds.yaml
  pipeline/
    assimilate.yaml
    reduce_validate.yaml
    optimize_conditions.yaml
```

- `defaults.yaml`: store/logging/seed等の共通デフォルト
- `sim/*`: backendと反応器モデル設定
- `task/*`: カテゴリ単体の設定
- `pipeline/*`: step列（taskを並べる）

> TODO: 既存リポジトリで `conf/` や `config/` が使われている場合、衝突を避けるため `configs/` に寄せる。

## 2. 上書き（override）規約
- CLI overrideは Hydra 標準に従う
  - 単発: `python -m <entry> sim=cantera_0d condition.T=700`
  - スイープ: `-m condition.T=700,750,800`
- “同じ条件が2箇所で定義される”状態を禁止（source of truthは1つ）

## 3. run dir / sweep dir
- `hydra.run.dir` は `artifacts/_hydra_runs/<timestamp>/<job_name>` のように、成果物（Artifact）とは分離する
- 解析成果物は `artifacts/` へ出す（hydra run dir に散らばらせない）

## 4. Seed
- `seed` は全カテゴリで共通キーにする（例: `common.seed`）
- 乱数を使う処理は必ず seed を参照する

## 5. 設定の固定（manifest化）
- 実行開始時、Hydraの最終設定（merge後）を `manifest.yaml` に保存する
- YAML内の相対パスは `repo_root` 起点で解決する

## 6. “入力編集→実行→出力→可視化” を YAMLで回す
- `sim` は単体でも `rxn sim run sim=...` のように実行可能
- `pipeline` では `steps` により同じRunArtifactを再利用しながら downstream を実行

