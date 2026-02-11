# 02_ARTIFACT_CONTRACTS（成果物契約）

本ドキュメントは、カテゴリ独立を成立させるための **Artifact（成果物）I/O契約** です。

## 1. 基本原則
- Artifactは「**manifest + data**」で構成する
- Artifactは原則 **イミュータブル**（更新は新Artifact）
- 上流/下流の結合は「Artifact契約」に限定する

## 2. 保存先（Storage layout）

### 2.1 推奨ディレクトリ
リポジトリ直下に `artifacts/` を持ち、種別ごとにサブフォルダを切る。

```
artifacts/
  _index.parquet               # 任意: 高速検索用（あとから追加）
  runs/
    <run_id>/
      manifest.yaml
      state.zarr/              # 主要データ（xarray.Dataset）
      logs.txt                 # 任意: 実行ログ
  run_sets/
    <run_set_id>/
      manifest.yaml
      runs.json                # case_id と run_id の対応（デバッグ/可視化用）
  observables/
    <obs_id>/
      manifest.yaml
      values.parquet           # run_id, observable, value, unit, meta
  graphs/
    <graph_id>/
      manifest.yaml
      graph.json               # 例: node/edge list
      stoich.npz               # 例: S行列（sparse）
  features/
    <feat_id>/
      manifest.yaml
      features.parquet
  sensitivity/
    <sens_id>/
      manifest.yaml
      sensitivity.parquet
  models/
    <model_id>/
      manifest.yaml
      model.bin                # 例: joblib/torch
  reduction/
    <red_id>/
      manifest.yaml
      mechanism_patch.yaml     # 元機構へ適用するパッチ
  validation/
    <val_id>/
      manifest.yaml
      metrics.parquet
  reports/
    <rep_id>/
      manifest.yaml
      index.html
```

> 現時点このリポジトリには `runs/` など既存ディレクトリが存在しない。外部既存リポジトリを統合する場合は、衝突回避のため `artifacts/` への集約を優先する（移行手順は別タスクで扱う）。

## 3. Artifact ID（命名）
- **再現性のため hash-based を推奨**
- `id = sha256(canonical_json(inputs + config + code_version)).hexdigest()[:16]`
- 人間が読むために任意の `tag` を manifest に持ってよい（IDには入れない）
- Hash input should be canonicalized (sorted keys, list/tuple normalized, Path to string, numpy scalars to Python).
- Transient config keys (e.g., logging/hydra) may be excluded from hashing via an explicit exclude list.

## 4. manifest.yaml（共通スキーマ）
最小必須フィールド（拡張可）:

```yaml
schema_version: 1
kind: runs | run_sets | observables | graphs | features | sensitivity | models | reduction | validation | reports
id: <artifact_id>
created_at: "2026-01-17T00:00:00Z"
parents: []          # 親artifact_id（派生関係）
inputs: {}           # 入力参照（例: run_id list, mechanism idなど）
config: {}           # 実行時の有効設定（Hydra最終形）
code:
  git_commit: "<optional>"
  dirty: false
  version: "<optional>"
provenance: {}       # 実行環境メタデータ（python/OS など）
notes: "<free text>"
```

## 5. RunArtifact（シミュ出力）契約

### 5.1 必須
- `state.zarr` は `xarray.Dataset` として読み出せること
- 少なくとも以下の coords を持つ:
  - `time`（定常でも長さ1で良い）
  - `species`（気相）または `surface_species`（表面）
  - `reaction`（反応ID）※反応がある場合
  - `phase`（gas/surface/other）※無い場合はattrsで説明
- `attrs` に単位系とモデル種別を持つ

### 5.2 推奨（下流の品質を上げる）
- `T`, `P`（time軸）
- `X`（mole fraction）, `C`（concentration）いずれか
- 反応寄与:
  - `rop_net` (time x reaction; phase optional)
  - `net_production_rates` (time x species; phase optional)
  - `creation_rates` / `destruction_rates` (time x species; phase optional)
- 表面:
  - `coverage`（surface_species×time）

### 5.3 “必要データ宣言”
下流の Task/Observable/Feature は「必要な変数」を宣言する。
足りない場合は **静かに無視しない**（例外 or 明確なwarn）。

## 5.4 RunSetArtifact（複数条件run束ね）契約

複数条件（CSV全件など）の run を「1つの入力参照」として下流に渡すためのArtifact。

### 必須
- kind: `run_sets`
- `manifest.inputs.run_ids`: 対象 run_id の配列（空でない）
- `manifest.inputs.case_ids`: 条件の識別子（`run_ids` と同順）
- `manifest.inputs.case_to_run`: `{case_id: run_id}` の対応表

### 推奨
- `runs.json`: 上記 inputs のコピーに加え、CSV行のメタ（T0/P0_atm/phi/t_end 等）を保持

## 6. ObservableArtifact（目的変数）契約

目的変数は多様（膜厚、組成、占有率、空間分布…）になり得る。
そこで2レベルの表現を許可する。

### 6.1 Scalar / Vector 観測（推奨）
- `values.parquet`（縦持ち）
  - columns: `run_id`, `observable`, `value`, `unit`, `meta_json`
- Parquet writer が無い環境では `values.json`（同一列定義）を併用して保存する場合がある。

### 6.2 High-dimensional 観測（将来）
- 画像/動画/空間分布は `zarr` とし、manifestに `shape`, `dtype`, `axes` を記録。

## 7. GraphArtifact（反応ネットワーク）契約
- 反応ネットワークは複数表現を併存して良い（相互変換可能な範囲で）
  - S行列（化学量論）
  - bipartite graph（species-reaction）
  - reaction-reaction graph
- 種ノード属性（元素組成、電荷、相、状態: radical/ion/neutral など）を可能な限り付与
- 反応属性（reaction type）を可能な限り付与

## 8. FeatureArtifact / SensitivityArtifact 契約
- `features.parquet` / `sensitivity.parquet` は DataFrame として読み出せる
- 必須列は以下のどれかを含む（用途で異なるため、すべて必須ではない）:
  - `run_id`, `condition_id`
  - `time_window`（集約区間がある場合）
  - `target`（膜厚など対象観測）
  - `reaction_id` / `species`
  - `value`, `unit`, `meta_json`
- Parquet writer が無い環境では `features.json` / `sensitivity.json`（同一列定義）を併用して保存する場合がある。

## 9. Reduction / Validation / Report
- 縮退は「パッチ」として保存（元機構を破壊しない）
- 妥当性評価は、対象パイプラインと許容誤差を manifest に持つ
