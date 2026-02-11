# Task: Docs更新（設計契約/運用/拡張ガイド）

## Summary
- **Goal**: 新構成に合わせてdocsを更新し、運用手順が迷わない状態にする。
- **Scope**: Invariants/Artifact契約/Config規約/拡張ガイド/Quickstart。
- **Non-goals**: 外部ツールの詳細導入手順。

## Depends on
- depends_on: T082R

## Contracts / References
- docs/00_INVARIANTS.md
- docs/02_ARTIFACT_CONTRACTS.md
- docs/03_CONFIG_CONVENTIONS.md

## Acceptance Criteria
- [ ] docs/README.md の「読む順」「最短手順」が更新される。
- [ ] 拡張ガイド（新しいreducer追加）が更新される。
- [ ] 出力規約とRunStore必須ファイルが明記される。

## Verification
Commands to run (examples):
- `python -m rxn_platform.cli help`

## Notes
- 既存の契約文書は破壊的に変更しない。
