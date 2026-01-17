# work/（Codex自動開発ループ用）

このディレクトリは **タスク管理** と **Codex自動実装ループ** のための運用資産です。

## 概要
- `queue.json` がタスクの真実（source of truth）
- 各タスクの詳細は `tasks/` にある MD（Acceptance Criteria + Verification 付き）
- `tools/codex_loop/run_loop.py` が `codex exec` を呼び、
  - 次のタスクを選択
  - Codexへ実装指示
  - Verificationを実行
  - queueの状態更新
  を行います

## 基本運用
1) タスクを追加/修正する場合
- `queue.json` と `tasks/<id>_*.md` を更新
- 契約変更が必要なら docs を更新（基本は避ける）

2) 自動実装を回す
- `python tools/codex_loop/run_loop.py --once`
- 連続実行: `python tools/codex_loop/run_loop.py`

> Codex CLI の `codex exec --full-auto` を前提とします（低フリクション自動化）。
> ただしネットワークやリポジトリ外操作は止まりやすいので、タスクで避ける。

