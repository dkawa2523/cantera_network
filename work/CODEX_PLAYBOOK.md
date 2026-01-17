# Codex運用プレイブック（止まらない自動ループのために）

## 前提
- `tools/codex_loop/run_loop.py` は `work/queue.json` を source of truth としてタスクを回します。
- 各タスクは `work/tasks/*.md` の **Acceptance Criteria** と **Verification** が合格するまで完了扱いになりません。

## よくある停止要因と回避策

### 1) 確認質問で止まる
- タスクMDにも wrapper にも「質問禁止」を明記済み。
- 不明点は **TODO** を残し、合理的仮定で進める。

### 2) 依存ライブラリ不足でテストが落ち続ける
- cantera / sbi などは **optional** とし、テストは `pytest.importorskip` を使う。
- それでも必須依存が足りない場合は doctor で検出できるようにする（T017）。

### 3) タスクが大きすぎて未完のまま次へ進む
- queue は“実装単位”に分割済み。
- もし still too big なら、`work/templates/TASK.md` からサブタスクを作って queue に追加する。

### 4) 前フェーズの品質が低いまま次フェーズに入り、破綻が連鎖する
- 本キットではフェーズ境界に **Phase Gate** を追加している。
  - P0→P1: `T020G`
  - P1→P2: `T040G`
  - P2→P3: `T052G`
  - P3→Release: `T060G`
- Gate は `pytest` と最小E2E（dummy中心、Canteraは任意）を実行し、
  **失敗したらそのタスク内で修正/軽リファクタまで完了**させる。

## 追加タスクの作り方（開発者向け）

1. `work/templates/TASK.md` をコピーして `work/tasks/T0xx_*.md` を作る  
2. `work/queue.json` にエントリ追加（depends_on を必ず指定）  
3. `python tools/codex_loop/run_loop.py --once` で実行

## 実行例

```bash
# 1タスクだけ
$ python tools/codex_loop/run_loop.py --once

# 連続実行
$ python tools/codex_loop/run_loop.py
```
