# T003 Core schema: Manifestモデル（pydantic）+ YAML I/O + schema_version

- **id**: `T003`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T002
- **unblocks**: (none)
- **skills**: python, pydantic

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

Artifact契約の核となる `manifest.yaml` のスキーマをコードに落とし込む。
後続タスク（Store/Backend/Observables/...）は必ずこのManifestモデルを使って保存する。

## Scope

- Pydantic（または dataclass + validation）で `ArtifactManifest` を定義する
- YAMLへの serialize/deserialize（read/write）を提供する
- schema_version / kind / id / created_at / parents / inputs / config / code / provenance を最低限扱う
- Manifestを「設定が真実」にするため、Hydraの有効設定を dict として格納できるようにする

## Out of scope

- ArtifactStoreの実装（T005で実施）
- RunArtifact固有のxarray仕様（T020/後続で検証）

## Deliverables

- `src/rxn_platform/core.py`（または `manifest.py`）に `ArtifactManifest` を追加
- `src/rxn_platform/core.py` に `load_manifest(path)` / `dump_manifest(path, manifest)`
- `tests/test_manifest_roundtrip.py`（YAML往復）

## Acceptance Criteria

- manifest を YAML へ書き出し→読み戻しで同値になる（最低限のフィールド）
- 未知フィールドの扱い方針（reject/allow）を明確化し、壊れたmanifestで落ち方が分かりやすい
- schema_version を必ず保存し、将来の移行に備える

## Verification

```bash
$ python -m compileall -q src
$ python -c "from rxn_platform.core import ArtifactManifest; m=ArtifactManifest(schema_version=1, kind='runs', id='x', created_at='2026-01-17T00:00:00Z', parents=[], inputs={}, config={}, code={}, provenance={}); print(m.kind)"
```

## Notes

- 依存追加は最小。pydanticが無い環境も考慮する場合は optional import にする（TODO可）。

## Final Response (Codex)

Codex の最終応答は **必ず** 次の JSON 形式（tools/codex_loop/response_schema.json）に従ってください。

- `status`: `done` / `blocked`
- `summary`: 何をしたか（短く）
- `files_changed`: 変更したファイル一覧
- `verification`: 実行したコマンドと結果
- `next`: 次にやるべきタスクID（推奨）

