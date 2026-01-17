# 01_ARCHITECTURE（現状→理想 / 依存方向）

## 1. 前提
- 本基盤は「シミュレーション（例: Cantera）を大量自動実行」し、
  反応ネットワークを考慮した **同化・最適化・縮退・可視化** を行う。
- ただし、Canteraに固定しない。将来別シミュレータに差し替え可能にする。

> TODO: 既存リポジトリ（dkawa2523/cantera_v1 等）の現状構造を反映した「現状図」は、実装開始後に追記。

## 2. ターゲットアーキテクチャ（理想）

### 2.1 レイヤ構造

```
+-----------------------------+
| CLI / Pipeline Runner       |
|  - hydra config             |
|  - step execution           |
+--------------+--------------+
               |
               v
+-----------------------------+
| Tasks (カテゴリ単位)        |
|  sim / observables / ...    |
|  graphs / features / ...    |
|  sensitivity / assimilation |
|  optimization / reduction   |
|  viz / bench                |
+--------------+--------------+
               |
               v
+-----------------------------+
| Artifact Store (I/O契約)    |
|  - immutable artifacts      |
|  - manifest + data          |
|  - index / registry         |
+--------------+--------------+
               ^
               |
+--------------+--------------+
| Simulation Backends         |
|  - CanteraBackend (one)     |
|  - Future backends          |
+-----------------------------+
```

### 2.2 依存方向（禁止事項も含む）
- ✅ 許可：Task → Artifact Store（読み/書き）
- ✅ 許可：Pipeline Runner → Task（起動）
- ✅ 許可：Backend → Artifact Store（RunArtifact出力）
- ❌ 禁止：Task A が Task B の内部関数を直接呼ぶ（スパゲッティ化）
- ❌ 禁止：Backend固有APIを解析カテゴリが直接参照する

## 3. 重要な設計判断（なぜこうするか）

### 3.1 “カテゴリ独立” の理由
- 同化と縮退は、同じRunArtifact/Graph/Featureを使うが、
  実装の結合を強くすると改修影響が大きくなる。
- 解析カテゴリの独立I/Oにより、
  - 新手法追加
  - 手法入れ替え
  - 組み合わせ変更（pipeline追加）
  が局所化する。

### 3.2 “Artifact契約” の理由
- 多条件・多反復でデータが膨大になる。
- 目的変数の追加や特徴量の追加を「再シミュなし」で回せることが重要。
- そのため、RunArtifactを資産化し、下流はRunArtifact再利用で回す。

## 4. 実装フェーズ（収束戦略）

### Phase 0（骨格）
- Artifact/Store/Task/Pipeline の最小実装
- Dummy backendで smoke test

### Phase 1（反応解析MVP）
- CanteraBackend
- observables（膜厚/組成/占有率）
- graphs/features/sensitivity の最小セット
- 縮退：閾値削除 + 回帰テスト
- 同化：ベースライン（CMA-ES） + EKI/ES-MDA いずれか1本

### Phase 2（高度化）
- DRGEP/DRGEPSAラッパ
- 多目的BO
- SBI / GNN（必要になったら）

