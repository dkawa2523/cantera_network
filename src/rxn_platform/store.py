"""Artifact storage with atomic write semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional, Union

from rxn_platform.core import ArtifactManifest, dump_manifest, load_manifest
from rxn_platform.errors import ArtifactError
from rxn_platform.run_store import normalize_component

_Payload = Union[str, bytes, bytearray, Path]


@dataclass(frozen=True)
class ArtifactCacheResult:
    path: Path
    reused: bool
    manifest: ArtifactManifest


def _coerce_parent_ids(
    parents: Optional[Iterable[Union[str, ArtifactManifest]]],
) -> list[str]:
    if parents is None:
        return []
    if isinstance(parents, (str, ArtifactManifest)):
        parents = [parents]
    parent_ids: list[str] = []
    for entry in parents:
        if isinstance(entry, ArtifactManifest):
            parent_id = entry.id
        elif isinstance(entry, str):
            parent_id = entry
        else:
            raise TypeError(
                "parents must contain strings or ArtifactManifest instances."
            )
        if not parent_id.strip():
            raise ValueError("parent ids must be non-empty strings.")
        parent_ids.append(parent_id)
    return parent_ids


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def apply_parents_inputs(
    manifest: ArtifactManifest,
    *,
    parents: Optional[Iterable[Union[str, ArtifactManifest]]] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    allow_input_overwrite: bool = False,
) -> ArtifactManifest:
    """Return a new manifest with merged parents/inputs for provenance tracking."""
    if not hasattr(manifest, "to_dict"):
        raise TypeError("manifest must be an ArtifactManifest instance.")
    payload = manifest.to_dict()
    merged_parents = _dedupe_preserve_order(
        list(payload.get("parents", [])) + _coerce_parent_ids(parents)
    )
    merged_inputs = dict(payload.get("inputs", {}))
    if inputs:
        for key, value in inputs.items():
            if key in merged_inputs and merged_inputs[key] != value:
                if not allow_input_overwrite:
                    raise ValueError(f"Input key already set: {key!r}")
            merged_inputs[key] = value
    payload["parents"] = merged_parents
    payload["inputs"] = merged_inputs
    return ArtifactManifest.from_dict(payload)


def _identity_payload(manifest: ArtifactManifest) -> dict[str, Any]:
    return {
        "kind": manifest.kind,
        "id": manifest.id,
        "inputs": manifest.inputs,
        "config": manifest.config,
        "code": manifest.code,
    }


def _assert_identity_match(
    expected: ArtifactManifest,
    actual: ArtifactManifest,
) -> None:
    if _identity_payload(expected) != _identity_payload(actual):
        raise ValueError("Existing artifact manifest does not match requested identity.")


class ArtifactStore:
    """Store artifacts under artifacts/<kind>/<id>/ with atomic writes."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)

    def artifact_dir(self, kind: str, artifact_id: str) -> Path:
        kind = normalize_component(kind, "kind")
        artifact_id = normalize_component(artifact_id, "artifact_id")
        return self.root / kind / artifact_id

    def manifest_path(self, kind: str, artifact_id: str) -> Path:
        return self.artifact_dir(kind, artifact_id) / "manifest.yaml"

    def exists(self, kind: str, artifact_id: str) -> bool:
        return self.manifest_path(kind, artifact_id).exists()

    def open_manifest(self, kind: str, artifact_id: str) -> ArtifactManifest:
        path = self.manifest_path(kind, artifact_id)
        if not path.exists():
            raise ArtifactError(f"Manifest not found: {path}")
        return load_manifest(path)

    def read_manifest(self, kind: str, artifact_id: str) -> ArtifactManifest:
        return self.open_manifest(kind, artifact_id)

    def ensure(
        self,
        manifest: ArtifactManifest,
        writer: Optional[Callable[[Path], None]] = None,
        *,
        compare_identity: bool = True,
    ) -> ArtifactCacheResult:
        """Return the cached artifact, or create it if missing."""
        if not hasattr(manifest, "to_dict"):
            raise TypeError("manifest must be an ArtifactManifest instance.")
        kind = normalize_component(manifest.kind, "kind")
        artifact_id = normalize_component(manifest.id, "artifact_id")
        target_dir = self.artifact_dir(kind, artifact_id)
        if self.exists(kind, artifact_id):
            existing = self.open_manifest(kind, artifact_id)
            if compare_identity:
                _assert_identity_match(manifest, existing)
            return ArtifactCacheResult(
                path=target_dir,
                reused=True,
                manifest=existing,
            )
        try:
            self.write_artifact_dir_atomic(manifest, writer=writer)
        except FileExistsError:
            existing = self.open_manifest(kind, artifact_id)
            if compare_identity:
                _assert_identity_match(manifest, existing)
            return ArtifactCacheResult(
                path=target_dir,
                reused=True,
                manifest=existing,
            )
        return ArtifactCacheResult(
            path=target_dir,
            reused=False,
            manifest=manifest,
        )

    def write_artifact_dir_atomic(
        self,
        manifest: ArtifactManifest,
        writer: Optional[Callable[[Path], None]] = None,
    ) -> Path:
        if not hasattr(manifest, "to_dict"):
            raise TypeError("manifest must be an ArtifactManifest instance.")
        kind = normalize_component(manifest.kind, "kind")
        artifact_id = normalize_component(manifest.id, "artifact_id")
        target_dir = self.artifact_dir(kind, artifact_id)
        if target_dir.exists():
            raise FileExistsError(f"Artifact already exists: {target_dir}")

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(
            tempfile.mkdtemp(prefix=f".{artifact_id}.", dir=target_dir.parent)
        )
        try:
            dump_manifest(tmp_dir / "manifest.yaml", manifest)
            if writer is not None:
                writer(tmp_dir)
            try:
                os.rename(tmp_dir, target_dir)
            except OSError as exc:
                if target_dir.exists():
                    raise FileExistsError(
                        f"Artifact already exists: {target_dir}"
                    ) from exc
                raise
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return target_dir

    def write_artifact(
        self,
        manifest: ArtifactManifest,
        data_files: Optional[Mapping[str, _Payload]] = None,
    ) -> Path:
        def _writer(base: Path) -> None:
            if not data_files:
                return
            for name, payload in data_files.items():
                self._write_payload(base, name, payload)

        return self.write_artifact_dir_atomic(manifest, writer=_writer)

    def _write_payload(self, base_dir: Path, name: str, payload: _Payload) -> None:
        rel_path = self._safe_relative_path(name)
        if rel_path.as_posix() == "manifest.yaml":
            raise ValueError("data_files may not overwrite manifest.yaml")
        target = base_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, Path):
            if payload.is_dir():
                shutil.copytree(payload, target)
            else:
                shutil.copy2(payload, target)
            return
        if isinstance(payload, (bytes, bytearray)):
            target.write_bytes(payload)
            return
        if isinstance(payload, str):
            target.write_text(payload, encoding="utf-8")
            return
        raise TypeError(f"Unsupported payload type for {name}: {type(payload)!r}")

    def _safe_relative_path(self, name: str) -> Path:
        path = Path(name)
        if not path.parts or path.parts == (".",):
            raise ValueError("data file name must be a relative path.")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"data file name must be relative: {name!r}")
        return path


__all__ = [
    "ArtifactCacheResult",
    "ArtifactStore",
    "apply_parents_inputs",
]
