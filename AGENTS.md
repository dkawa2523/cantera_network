# Codex運用: 永続ルール（このリポジトリ全体で常に有効）

このファイルは Codex CLI / IDE 拡張により自動読み込みされます。以後の全タスク実装は **docs/** の契約に従ってください。

## 0. 最優先で守るもの
1) `docs/00_INVARIANTS.md`（不変条件）
2) `docs/02_ARTIFACT_CONTRACTS.md`（Artifact 契約）
3) `docs/03_CONFIG_CONVENTIONS.md`（Hydra 設定規約）
4) `work/tasks/*.md`（各タスクの Acceptance Criteria / Verification）

## 1. 自動化前提（停止しない）
- **確認質問をしない**：不明点があっても、合理的な仮定で前に進み、必要なら `TODO:` を残す。
- **#approve / 追加承認待ちを作らない**：実装・テスト・修正をタスク完了まで継続する。
- **ネットワークに依存しない**：`pip install` や外部DLを前提にしない（必要なら `TODO` として明示）。
- **作業ディレクトリ外を書き換えない**：リポジトリ外へ触れる設計は避ける。

> 自動実行は `codex exec --full-auto` を想定。`--full-auto` は `workspace-write` + `on-request approvals` のプリセットです。追加の危険権限（`--yolo`/`danger-full-access`）は、隔離環境でのみ使用。

## 2. タスク実行の基本手順（各回共通）
1) 対象タスクMDを読む（`work/tasks/<TASK_ID>_*.md`）
2) `docs/README.md` から関連規約を確認
3) 既存コードを最小限に観察（探索）し、差分を最小に実装
4) タスクの **Acceptance Criteria** を満たすまで実装
5) タスクの **Verification** コマンドを実行し、失敗なら直す
6) 最終出力は、（自動ループの場合）`tools/codex_loop/response_schema.json` に適合するJSONで返す

## 3. 実装ポリシー（スパゲッティ防止）
- **カテゴリ間はArtifactで接続**：カテゴリ同士を関数呼び出しで直接依存させない。
- **「便利関数」増殖を避ける**：共通化したくなったら、まず契約（Artifact/Task API）を見直す。
- **小さく完結**：タスクは「完了=検証コマンドが通る」まで。
- **最小ファイル数**：新規モジュール増殖より、既存のカテゴリファイルにプラグイン登録を優先。
- **型とスキーマ**：I/O は Pydantic などで明示し、破壊的変更を避ける。

## 4. 返答フォーマット（自動ループ用）
自動ループ（`tools/codex_loop/run_loop.py`）では `codex exec --output-schema tools/codex_loop/response_schema.json` を使います。
**最終メッセージは必ず JSON のみ**を返し、schemaに適合させること。

（対話UIでの実行時はMarkdownでもよいが、最後に同等情報をJSONでも付与することを推奨。）
