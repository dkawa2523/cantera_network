# T020G Phase Gate P0→P1: P0全体の仕様準拠・統合テスト・修正/軽リファクタ

- **id**: `T020G`
- **priority**: `P0`
- **status**: `todo`
- **depends_on**: T001..T020（queue.json参照）
- **unblocks**: P1一式（T021..T040）
- **skills**: qa, integration, refactor

## Contracts (must follow)

- `docs/00_INVARIANTS.md`
- `docs/02_ARTIFACT_CONTRACTS.md`
- `docs/03_CONFIG_CONVENTIONS.md`
- `docs/06_TESTING_AND_QA.md`

## Background

P0（基盤）は後続すべての開発の土台であり、ここで仕様ズレやI/O契約違反が残ると、
P1以降で “原因不明の破綻” が連鎖します。

そこで P0 完了時点で、
- docs契約（Invariants / Artifact / Hydra規約）に対する整合
- 最小機能（dummy backend）でのE2E実行
- 例外・ログ・CLI入口
を **統合テスト** し、問題があればこのタスク内で修正・軽リファクタまで完了させます。

## Scope

### テスト（必須）
- `pytest` の全テストが通ること
- `rxn_platform` が import できること
- CLIの help と doctor が動くこと
- dummy backend で smoke pipeline が実行でき、artifacts が生成されること

### 仕様準拠チェック（必須）
- Artifact layout / manifest の最低要件が満たされていること
- `run_id/artifact_id` が安定生成されること（同設定で同ID）
- immutability（既存artifactを上書きしない）

### 修正/軽リファクタ（必要なら）
- 失敗原因を特定し、最小変更で修正
- 依存方向（カテゴリ間直接依存）や “便利関数増殖” が起きていたら抑止
- テストを追加/修正して再発を防ぐ（ただし過剰に増やさない）

## Out of scope

- Cantera を使った重い統合テスト（P1ゲートで扱う）
- 反応ネットワークや特徴量の正しさ（P1ゲートで扱う）

## Deliverables

- （必要なら）tests の追加/修正
- （必要なら）CLI/runner/store のバグ修正
- （必要なら）docs の TODO 埋め（捏造は禁止）

## Acceptance Criteria

- P0 の Verification がすべて成功する
- “仕様の真実=Hydra config” が守られ、実行時設定が manifest に残る
- dummy smoke が動き、RunArtifact が契約に沿う最低限の内容を持つ
- 修正が必要だった場合、**再発防止の最小テスト**が追加されている

## Verification

```bash
$ python -m compileall -q src
$ pytest -q
$ python -m rxn_platform.cli --help
$ python -m rxn_platform.cli doctor
$ python -m rxn_platform.cli pipeline run pipeline=smoke sim=dummy
```

## Notes

- doctor / pipeline コマンド名が実装とズレている場合は、このタスク内で **CLI規約に合わせて統一** する。
- どうしても環境依存（OS/権限）で失敗する場合は、doctor が理由を明示し、テスト側はskipにする。

