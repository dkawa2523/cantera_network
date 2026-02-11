# tools/codex_loop（Codex自動実装ループ）

このツールは `work/queue.json` を読み、次の todo タスクを `codex exec` に渡して実装を進めます。

## 前提
- `codex` CLI がインストールされていること
- 実行は **リポジトリルート**（`work/queue.json` がある場所）で行う

## 使い方

### 1) 1タスクだけ実行
```bash
$ python tools/codex_loop/run_loop.py --once
```

### 2) 連続で実行（runnable な todo が無くなるまで）
```bash
$ python tools/codex_loop/run_loop.py
```

### 3) 特定タスクだけ
```bash
$ python tools/codex_loop/run_loop.py --task-id T004 --once
```

### 4) キュー確認
```bash
$ python tools/codex_loop/run_loop.py --list
```

## ログ
- `work/logs/<task_id>_<timestamp>/` に以下が保存されます
  - `prompt.md`（Codexに渡したプロンプト）
  - `codex_stdout.txt` / `codex_stderr.txt`
  - `codex_last_message.json`（最終応答。JSON schema 検証済み）
  - `verification.log`（Verificationコマンド出力）

## 自動化のポイント
- 原則 `--full-auto` を使う（承認待ちを最小化）
- ネットワークやリポジトリ外操作は止まりやすいので、タスクで避ける
- 出力は `tools/codex_loop/response_schema.json` に合わせてJSONのみ
- 同時実行は避ける（`work/.codex_loop.lock` で排他制御）

## 危険オプション
- `--yolo` は承認・sandboxをバイパスする可能性があります。
  **隔離環境（使い捨てコンテナ等）でのみ** 使ってください。
