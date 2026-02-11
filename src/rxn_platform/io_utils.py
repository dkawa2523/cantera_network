"""Shared JSON/YAML I/O helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Type

try:  # Optional dependency.
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _strip_inline_comment(line: str) -> str:
    if "#" not in line:
        return line
    # Cheap YAML comment stripping: sufficient for our config subset.
    return line.split("#", 1)[0].rstrip()


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_square = 0
    depth_curly = 0
    quote: Optional[str] = None
    escaped = False
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if quote is not None:
            buf.append(ch)
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            continue
        if ch == "[":
            depth_square += 1
            buf.append(ch)
            continue
        if ch == "]":
            depth_square = max(0, depth_square - 1)
            buf.append(ch)
            continue
        if ch == "{":
            depth_curly += 1
            buf.append(ch)
            continue
        if ch == "}":
            depth_curly = max(0, depth_curly - 1)
            buf.append(ch)
            continue
        if ch == sep and depth_square == 0 and depth_curly == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


def _split_flow_mapping_entry(entry: str) -> tuple[str, str]:
    depth_square = 0
    depth_curly = 0
    quote: Optional[str] = None
    escaped = False
    for idx, ch in enumerate(entry):
        if escaped:
            escaped = False
            continue
        if quote is not None:
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == "[":
            depth_square += 1
            continue
        if ch == "]":
            depth_square = max(0, depth_square - 1)
            continue
        if ch == "{":
            depth_curly += 1
            continue
        if ch == "}":
            depth_curly = max(0, depth_curly - 1)
            continue
        if ch == ":" and depth_square == 0 and depth_curly == 0:
            return entry[:idx], entry[idx + 1 :]
    raise ValueError(f"Invalid flow mapping entry (missing ':'): {entry!r}")


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip()
    if not cleaned:
        return ""

    # Flow-style collections (single-line only).
    if cleaned.startswith("[") and cleaned.endswith("]"):
        inner = cleaned[1:-1].strip()
        if not inner:
            return []
        parts = _split_top_level(inner, ",")
        return [_parse_scalar(part) for part in parts if part.strip()]
    if cleaned.startswith("{") and cleaned.endswith("}"):
        inner = cleaned[1:-1].strip()
        if not inner:
            return {}
        parts = _split_top_level(inner, ",")
        mapping: dict[str, Any] = {}
        for part in parts:
            if not part.strip():
                continue
            key_raw, value_raw = _split_flow_mapping_entry(part)
            key = _parse_scalar(key_raw)
            if not isinstance(key, str):
                key = str(key)
            mapping[key] = _parse_scalar(value_raw)
        return mapping

    lowered = cleaned.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if cleaned[0] in {"'", '"'} and cleaned[-1] == cleaned[0]:
        return cleaned[1:-1]

    # Numbers (int/float/scientific).
    try:
        if lowered.isdigit() or (
            cleaned.startswith(("-", "+")) and cleaned[1:].isdigit()
        ):
            return int(cleaned)
        return float(cleaned)
    except ValueError:
        return cleaned


def _split_key_value(line: str, *, path: Path, line_no: int) -> tuple[str, Optional[str]]:
    if ":" not in line:
        raise ValueError(f"Invalid YAML line in {path} at {line_no}: {line}")
    key, value = line.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Empty YAML key in {path} at {line_no}.")
    value = value.strip()
    if value == "":
        return key, None
    return key, value


def _parse_block(
    lines: list[str],
    index: int,
    indent: int,
    *,
    path: Path,
) -> tuple[Any, int]:
    while index < len(lines) and _is_blank(lines[index]):
        index += 1
    if index >= len(lines):
        return None, index
    line = _strip_inline_comment(lines[index])
    line_indent = _line_indent(line)
    if line_indent < indent:
        return None, index
    if line_indent > indent:
        raise ValueError(f"Unexpected indent in {path} at line {index + 1}.")
    stripped = line.strip()
    if stripped.startswith("- "):
        return _parse_list(lines, index, indent, path=path)
    return _parse_mapping(lines, index, indent, path=path)


def _parse_mapping(
    lines: list[str],
    index: int,
    indent: int,
    *,
    path: Path,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(lines):
        if _is_blank(lines[index]):
            index += 1
            continue
        line = _strip_inline_comment(lines[index])
        line_indent = _line_indent(line)
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indent in {path} at line {index + 1}.")
        stripped = line.strip()
        if stripped.startswith("- "):
            raise ValueError(f"Unexpected list item in {path} at line {index + 1}.")
        key, value = _split_key_value(stripped, path=path, line_no=index + 1)
        if value is None:
            nested, index = _parse_block(lines, index + 1, indent + 2, path=path)
            mapping[key] = nested
        else:
            mapping[key] = _parse_scalar(value)
            index += 1
    return mapping, index


def _parse_list(
    lines: list[str],
    index: int,
    indent: int,
    *,
    path: Path,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        if _is_blank(lines[index]):
            index += 1
            continue
        line = _strip_inline_comment(lines[index])
        line_indent = _line_indent(line)
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indent in {path} at line {index + 1}.")
        stripped = line.strip()
        if not stripped.startswith("- "):
            break
        content = stripped[2:].strip()
        if not content:
            item, index = _parse_block(lines, index + 1, indent + 2, path=path)
            items.append(item)
            continue
        if (content.startswith("{") and content.endswith("}")) or (
            content.startswith("[") and content.endswith("]")
        ):
            items.append(_parse_scalar(content))
            index += 1
            continue
        if ":" in content:
            key, value = _split_key_value(content, path=path, line_no=index + 1)
            item: dict[str, Any] = {}
            if value is None:
                nested, index = _parse_block(lines, index + 1, indent + 2, path=path)
                item[key] = nested
            else:
                item[key] = _parse_scalar(value)
                index += 1
            while True:
                next_index = index
                while next_index < len(lines) and _is_blank(lines[next_index]):
                    next_index += 1
                if next_index >= len(lines):
                    break
                next_indent = _line_indent(lines[next_index])
                if next_indent <= indent:
                    break
                if next_indent < indent + 2:
                    raise ValueError(
                        f"Unexpected indent in {path} at line {next_index + 1}."
                    )
                extra, index = _parse_mapping(lines, next_index, indent + 2, path=path)
                item.update(extra)
                break
            items.append(item)
            continue
        items.append(_parse_scalar(content))
        index += 1
    return items, index


def _simple_yaml_load(text: str, *, path: Path) -> Any:
    lines = text.splitlines()
    payload, _ = _parse_block(lines, 0, 0, path=path)
    if payload is None:
        return {}
    return payload


def read_yaml_payload(
    path: Path,
    *,
    error_message: Optional[str] = None,
    error_cls: Type[Exception] = ValueError,
) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return _simple_yaml_load(text, path=path)
        except Exception as exc:
            if error_message:
                raise error_cls(error_message) from exc
            raise


def write_yaml_payload(
    path: Path,
    payload: Mapping[str, Any],
    *,
    sort_keys: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if yaml is None:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=sort_keys,
                ensure_ascii=True,
            )
            handle.write("\n")
            return
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=False,
            default_flow_style=False,
            sort_keys=sort_keys,
        )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_handle: Optional[int] = None
    tmp_path: Optional[str] = None
    try:
        tmp_handle, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(tmp_handle, "w", encoding="utf-8") as handle:
            tmp_handle = None
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_handle is not None:
            try:
                os.close(tmp_handle)
            except OSError:
                pass
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


__all__ = [
    "read_json",
    "read_yaml_payload",
    "write_json_atomic",
    "write_yaml_payload",
]
