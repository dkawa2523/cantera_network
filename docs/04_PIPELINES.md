# 04_PIPELINES（パイプライン規約）

パイプラインは **YAML宣言**で定義し、コード側は「実行器」に徹します。

## 1. pipeline YAML スキーマ（最小）

```yaml
steps:
  - id: obs
    task: observables.compute
    inputs:
      runs: ${run_ids}
    params:
      observables: [film_thickness, gas_composition]

  - id: sens
    task: sensitivity.multiplier_scan
    inputs:
      runs: ${run_ids}
    params:
      targets: [film_thickness]
      delta: 0.05

  - id: assim
    task: assimilation.eki
    inputs:
      runs: ${run_ids}
      observables: @obs
      sensitivity: @sens
    params:
      ensemble_size: 64

  - id: rep
    task: viz.make
    inputs:
      runs: ${run_ids}
      model: @assim
    params:
      dashboard: ds
```

## 2. 参照記法
- `@<id>`: 指定stepの出力Artifactを参照
- `@last`: 直前stepの出力Artifactを参照
- `${var}`: Hydra interpolation

> 注意: Step出力は「Artifact ID」1つを基本にし、複数出力が必要なら `outputs:` を増やす（TODO:実装時に確定）。

## 3. 代表パイプライン（推奨）

### 3.1 assimilation（同化）
- sim → observables → sensitivity → assimilation → viz

### 3.2 reduce_validate（縮退→妥当性評価）
- sim → features/importance → reduction → sim(reduced) → validation → viz

### 3.3 optimize_conditions（条件最適化）
- initial sampling → sim → observables → optimizer propose → sim ...（ループ）

> ループ制御は最初は “外側スクリプト” に逃がし、パイプライン自体は逐次でもよい（P0）。

## 4. パイプライン追加手順
1) `configs/pipeline/<name>.yaml` を作る
2) 既存Taskキー（`task:`）を組み合わせる
3) 依存Artifactの型が合うか（契約）を確認
4) `work/bench` の smoke test へ最小ケースを追加

