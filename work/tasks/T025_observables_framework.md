# T025 Observables framework: ObservableプラグインAPI + ObservableArtifact出力

- **id**: `T025`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T012, T020
- **unblocks**: (none)
- **skills**: python, domain-chemistry

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

膜厚/組成/占有率など“目的変数”は後から増減するため、観測量は Observables としてプラグイン化する。
同化/最適化/縮退評価は Observables を共通インターフェースで参照する。

## Scope

- Observable base（compute(run_dataset, cfg) -> DataFrame/Dict）を定義
- Observable registry を使って複数Observableを一括計算する task（例: observables.run）を実装
- 出力は ObservableArtifact（table + meta）として保存
- 複数目的変数に対応するため、出力は “name付きのベクトル” を基本にする（列名規約）

## Out of scope

- 観測値と物理モデル（表面反応の高度モデル）の確定（段階的）

## Deliverables

- `src/rxn_platform/tasks/observables.py`（framework + task）
- `tests/test_observables_framework.py`（dummy runでOK）
- `configs/observable/*.yaml`（雛形）

## Acceptance Criteria

- Observableを追加登録すると、observables.runで一括計算できる
- 欠損入力（変数が無い）でも graceful に NaN/skip と理由を出せる
- ObservableArtifactに入力run_idが記録される

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

