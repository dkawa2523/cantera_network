# T011 Backend IF + DummyBackend: SimulationBackend差し替え点と超高速ダミー実装

- **id**: `T011`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T005, T007
- **unblocks**: (none)
- **skills**: python, simulation

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

Backend差し替え点（SimulationBackend）を固定し、Canteraが無い環境でも回る DummyBackend を用意する。
これにより、後続のgraphs/features/viz/assim/opt/reductionの多くをdummyでテストできる。

## Scope

- `SimulationBackend` インターフェースを定義（入力cfg→RunArtifact相当のxarray Dataset）
- DummyBackend を実装し、決定論的に time/T/P/X/（任意でrop/wdot）を生成
- DummyBackendを registry に登録（例: backend.dummy）
- Taskとして `sim.run`（backendを呼びRunArtifactをArtifactStoreへ保存）を実装し登録

## Out of scope

- Cantera backend（T021で実施）
- 並列化（後回し）

## Deliverables

- `src/rxn_platform/backends/base.py`（IF）
- `src/rxn_platform/backends/dummy.py`
- `src/rxn_platform/tasks/sim.py`（sim.run task）
- `tests/test_dummy_backend_runartifact.py`

## Acceptance Criteria

- dummy backend で RunArtifact（manifest + state）相当が生成できる
- sim.run task が registry から解決できる
- Canteraが無い環境でもテストが通る

## Verification

```bash
$ python -m compileall -q src
```

## Notes

- RunArtifactの保存形式（zarr/nc）は後タスクで確定。ここでは最小でOK。

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

