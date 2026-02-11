# 09_REFACTOR_PLAN（大規模リファクタ計画）

本計画は「設定と出力の簡素化」「RunStore一本化」「プラグイン化によるスパゲッティ抑止」を段階的に行うためのロードマップです。
実装は work/tasks/ に分割し、P0→P2の順で安全に進めます。

## 目的（再掲）
- 設定（Hydra）を default + recipes に縮退
- RunStoreの出力規約を唯一の真実にする
- 手法追加のif/else増殖を廃止し、plugin registry 経由に統一
- 解析で使える図に可視化を刷新

## YAML削減プラン（最初の一手）
1. `configs/default.yaml` を唯一の入口とし、`configs/recipe/*.yaml` を用途別に最小限維持
2. legacyグループ（sim/task/pipeline/observable など）は互換読み込みに留め、徐々に置換
3. structured config（dataclass）を真実とし、yamlは上書きのみ
4. recipeから「RunStore出力規約」「RunStore必須ファイル」へ自動的に繋がるように統一

## 出力一本化プラン
- すべてのrunは `runs/<exp>/<run_id>/` に保存
- 必須: `manifest.json`, `config_resolved.yaml`, `metrics.json`
- 時系列: `sim/timeseries.zarr`
- グラフ: `graphs/species_graph/layer_*.npz` + `graphs/meta.json`
- 可視化: `viz/` 配下（最大3カテゴリ）

## 優先度
- P0: RunStore/設定/CLIの基盤整理
- P1: 可視化刷新と手法のプラグイン統合
- P2: 回帰テスト・ドキュメント更新

## 対象タスク
- 詳細は `work/queue.json` と `work/tasks/` を参照
