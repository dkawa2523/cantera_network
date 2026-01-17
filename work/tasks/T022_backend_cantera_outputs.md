# T022 Cantera outputs拡張: wdot/rop_net/creation/destruction 等を可能な範囲で保存

- **id**: `T022`
- **priority**: `P1`
- **status**: `todo`
- **depends_on**: T021, T020
- **unblocks**: (none)
- **skills**: python, cantera

## Contracts (must follow)

- `docs/02_ARTIFACT_CONTRACTS.md`

## Background

同化/最適化/縮退では、濃度だけでなく ROP や生成消滅寄与が重要な特徴量となる。
RunArtifactに保存できる出力を増やし、後続のfeatures/graphsへ渡す。

## Scope

- 可能なら net production rate (wdot) を species ごとに保存
- 可能なら reaction progress rate (rop) を reaction ごとに保存（net/forward/reverse のうち少なくとも1つ）
- creation/destruction の分解が取れるなら保存（取れない場合は TODO/skip）
- 変数名/単位を docs の契約に沿って整理（最低限 meta に units）

## Out of scope

- 全ての派生量の完全な計算（段階的）

## Deliverables

- `src/rxn_platform/backends/cantera.py` の拡張
- `tests/test_cantera_outputs_optional.py`（optional）
- docs/02_ARTIFACT_CONTRACTS.md の RunArtifact項目を必要なら追記

## Acceptance Criteria

- RunArtifactに追加変数が入る（cantera環境）
- 出力が無い場合も壊れず、features側で欠損を扱える
- 変数の軸（time×species / time×reaction）が一貫している

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

