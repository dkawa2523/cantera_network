# 07_CONTRIBUTING（開発規約）

## 1. コーディング規約（最小）
- Python は型ヒントを基本（mypyは任意だが推奨）
- I/O境界（Artifact, Config, Task）には Pydantic 等でスキーマを置く
- 例外は握りつぶさず、原因がわかるメッセージにする

## 2. 依存追加のルール
- 依存は増やしすぎない（レビュー負荷と運用負荷の増大）
- どうしても必要な場合は docs に理由を書く
- ネットワーク前提のインストール手順は最小化し、可能なら lockfile を使う

## 3. ディレクトリ規約
- スパゲッティ化防止のため「カテゴリ=1モジュール」から開始
- 手法が増える場合は plugin registry で増やす（サブパッケージ深掘りは最後）

## 4. ドキュメント更新
- 契約変更は原則禁止（docs/00...）
- どうしても必要なら「互換性維持」「移行手順」「影響範囲」を必ず書く

## 5. タスク追加・更新の手順
- `work/queue.json` と `work/tasks/Txxx_*.md` をセットで更新する（Acceptance Criteria / Verification を必ず記載）。
- 必要に応じて `work/STATUS.md` の件数と next を更新する。
- 新しい契約ドキュメントを追加した場合は `docs/README.md` の読む順に追記する。
- CI を追加する場合は `pytest -q` が短時間で通ることを前提にし、ネットワーク依存が避けられない場合は TODO として明記する。

## 6. スパゲッティ防止のガードレール
- カテゴリ間の接続は Artifact のみ（直接関数呼び出しをしない）。
- 便利関数を増やす前に、Artifact 契約や Task API の見直しで吸収できないか検討する。
- 新しい Artifact 種別や必須フィールドを追加した場合は、`docs/02_ARTIFACT_CONTRACTS.md` と validator/テストも更新する。

## 7. 縮退法（reducer）追加手順（Plugin registry 前提）
- 既存の `src/rxn_platform/tasks/reduction.py` に実装を追加する（ファイル増殖を避ける）。
- 関数は `cfg`（または `config`）を受け取り、`ArtifactCacheResult` を返す。
- 生成物は `docs/02_ARTIFACT_CONTRACTS.md` に沿って `reduction/<id>/` 配下へ保存する。
- Plugin registry に登録する:
  - `register("task", "reduction.<your_name>", <your_function>)`
  - `__all__` に追加して公開する。
- 入口は `reduction.dispatch` を使い、`method=<your_name>` で選択する（if/else増殖禁止）。
- 必要に応じて pipeline を追加する:
  - `configs/pipeline/reduce_<your_name>.yaml` を作成し `task: reduction.dispatch` と `method: <your_name>` を使う。
  - 入口が必要なら `configs/recipe/reduce_<your_name>.yaml` も用意する（recipe は短く）。
- 新規ファイルを作る場合は `src/rxn_platform/tasks/runner.py` の `_load_builtin_plugins()` に import を追加する。
- 追加した縮退法の smoke / regression テストを `tests/` に足す（重要度に応じて）。
- `benchmark_compare` は registry に登録された `reduction.*` タスクから reducer を推定する。

## 8. Lint / format / type / pre-commit
- 開発用依存（オフライン環境は社内ミラー/ローカル wheel を使用）:
  - `python -m pip install -e ".[dev]"`（TODO: オフライン手順を追記）
- Lint / format / type:
  - `ruff check .`
  - `black --check .`
  - `mypy`
- pre-commit（ローカル hook / ネットワーク不要）:
  - `pre-commit install`
  - `pre-commit run --all-files`
