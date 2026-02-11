# T057 SBI interface skeleton: sbiがあれば事後推定、無ければskip（summary=features）

- **id**: `T057`
- **priority**: `P3`
- **status**: `todo`
- **depends_on**: T034, T025
- **unblocks**: (none)
- **skills**: python, bayesian-inference

## Contracts (must follow)

- `docs/00_INVARIANTS.md`

## Background

SBI（Simulation-based inference）はブラックボックスシミュレータに適用できるが依存が重い場合がある。
sbi があれば実行、無ければskipする形でインターフェースだけ用意する。

## Scope

- simulation function: parameter -> observable/features を返す関数を作る（pipeline利用）
- summary statistics として FeatureArtifact を利用する
- sbi がインストールされている場合は最小のSNPE等を走らせる（小規模）
- 無い場合は明示的に skip し、手順をdocsにTODOとして残す

## Out of scope

- 高度なニューラルサロゲート（NeuralODE等）は対象外

## Deliverables

- `src/rxn_platform/tasks/sbi.py`（新規）
- `tests/test_sbi_optional_import.py`（importorskip）

## Acceptance Criteria

- sbi無し環境でも壊れない
- sbi有りなら最小実行ができる（軽量）

## Verification

```bash
$ pytest -q
```

## Notes

- (none)

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

