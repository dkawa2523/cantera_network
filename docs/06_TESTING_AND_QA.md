# 06_TESTING_AND_QA（テスト/品質/回帰）

本基盤は「多条件自動実行」「縮退」「同化」「最適化」を安全に回すため、
**回帰テスト（golden suite）** を重視します。

## 1. テストの種類

### 1.1 Unit tests（関数単位）
- 可能な限り純粋関数（副作用なし）にして unit test を書く
- 例：
  - S行列生成
  - 反応タイプ分類
  - Artifact id hash の安定性

### 1.2 Integration tests（Task単位）
- Dummy backend（超高速）で以下が通ることを保証
  - sim → observables → graphs → features → viz の最小パイプライン
- Cantera backend を使うテストは重くなりがちなので、最小ケースだけに限定

### 1.3 Regression tests（golden suite）
- 代表条件（少数）を固定し、以下を比較する
  - 目的変数（膜厚/組成/占有率）
  - 主要ROPランキング
  - 縮退前後の誤差が許容範囲内か
- 許容誤差は config で明示し、manifest に記録
  - 例: `configs/golden/dummy.yaml` に期待値と許容誤差を定義し、`tests/test_golden_suite_dummy.py` で検証

## 2. doctor コマンド（運用）
- `doctor` は環境・設定・最小実行ができることを検査する
- 例：
  - Python 依存の存在確認
  - Hydra config の整合性チェック
  - ダミーパイプライン実行

## 3. CI（最小）
- 少なくとも以下は自動で回ることを目標
  - lint / format（任意）
  - unit tests
  - dummy integration test

> 現時点このリポジトリには CI 設定（GitHub Actions等）が存在しない。外部既存リポジトリがある場合は状況を確認し、最小の追加で統合する。
