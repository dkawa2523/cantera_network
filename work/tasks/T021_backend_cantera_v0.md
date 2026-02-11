# T021 CanteraBackend v0: 0D実行（T/P/X, time grid）でRunArtifact生成

- **id**: `T021`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T011, T010
- **unblocks**: (none)
- **skills**: python, cantera, xarray

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

Canteraが使える環境では、0D反応計算を自動実行して RunArtifact として保存できる必要がある。
一方で、Canteraが無い環境でも基盤が壊れないよう optional に扱う。

## Scope

- CanteraSolutionを読み込み（mechanism yaml/cti）、初期条件（T/P/X）と反応器条件を設定して時間発展を計算
- 出力を xarray Dataset としてまとめ、RunArtifact として ArtifactStore に保存
- 設定（time_grid, reactor_type, atol/rtol, max_steps等）をHydra configで指定可能にする
- Cantera未導入なら backend 登録はするが実行時に分かりやすくエラー or skip できる

## Out of scope

- 表面反応/プラズマのフル対応（段階的に追加）
- 超高速化（後回し）

## Deliverables

- `src/rxn_platform/backends/cantera.py`（CanteraBackend）
- `tests/test_cantera_backend_optional.py`（cantera無い場合skip）
- `configs/sim/cantera_min.yaml`（最小例）

## Acceptance Criteria

- cantera がある環境で gri30 等の小さな機構でRunArtifactが生成できる
- cantera が無い環境ではテストがskipされ、他機能は壊れない
- 出力Datasetに time 次元と、少なくとも T,P と主要種のXが入る（可能な範囲）

## Verification

```bash
$ pytest -q
```

## Notes

- 0Dの選択（IdealGasConstPressureReactor 等）は config で切替可能にする。

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

