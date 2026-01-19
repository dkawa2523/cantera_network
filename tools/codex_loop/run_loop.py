#!/usr/bin/env python3
"""Codex 自動実装ループ

目的:
- work/queue.json を単一の真実とし、次の todo タスクを選んで codex exec に渡す
- codex の最終出力(JSON)を保存
- タスクMDの Verification コマンドを実行し、合格なら done に更新

注意:
- 本スクリプトは「リポジトリ外操作」「ネットワーク」等を避ける前提
- codex CLI が別途インストールされている必要があります
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


QUEUE_PATH = Path("work/queue.json")
STATE_PATH = Path("work/state.json")
STATUS_MD = Path("work/STATUS.md")
LOGS_DIR = Path("work/logs")
WRAPPER_PATH = Path("tools/codex_loop/prompt_wrapper.md")
SCHEMA_PATH = Path("tools/codex_loop/response_schema.json")


@dataclass
class VerificationResult:
    commands: List[str]
    passed: bool
    output: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / QUEUE_PATH).exists():
            return p
    raise FileNotFoundError(f"Could not find repo root containing {QUEUE_PATH} starting from {start}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_state(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / STATE_PATH
    if not path.exists():
        return {"schema_version": 1, "tasks": {}}
    return load_json(path)


def update_state_task(state: Dict[str, Any], task_id: str, **kwargs: Any) -> None:
    tasks = state.setdefault("tasks", {})
    task_state = tasks.setdefault(task_id, {"attempts": 0})
    task_state.update(kwargs)


def load_queue(repo_root: Path) -> List[Dict[str, Any]]:
    path = repo_root / QUEUE_PATH
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("queue.json must be a JSON list")
    return data


def save_queue(repo_root: Path, queue: List[Dict[str, Any]]) -> None:
    save_json(repo_root / QUEUE_PATH, queue)


def tasks_by_id(queue: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t["id"]: t for t in queue}


def deps_done(task: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> bool:
    for dep in task.get("depends_on", []) or []:
        if dep not in by_id:
            return False
        if by_id[dep].get("status") != "done":
            return False
    return True


def pick_next_task(queue: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    by_id = tasks_by_id(queue)
    candidates = [
        t
        for t in queue
        if t.get("status") == "todo" and deps_done(t, by_id)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda t: (int(t.get("priority", 999)), t.get("id", "")))
    return candidates[0]


VERIF_HEADER_RE = re.compile(r"^##\s+Verification\s*$")
CODE_FENCE_RE = re.compile(r"^```")
SHELL_CMD_RE = re.compile(r"^\$\s+(.*)\s*$")


def extract_verification_commands(task_md: str) -> List[str]:
    """Extract shell commands from the '## Verification' section.

    We keep it intentionally simple:
    - Find the '## Verification' header
    - Collect lines like '$ <command>' within subsequent fenced code blocks
    """
    lines = task_md.splitlines()
    in_verif = False
    in_fence = False
    commands: List[str] = []

    for line in lines:
        if not in_verif:
            if VERIF_HEADER_RE.match(line.strip()):
                in_verif = True
            continue

        # After entering Verification section
        if line.startswith("## ") and not VERIF_HEADER_RE.match(line.strip()):
            # next section
            break

        if CODE_FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue

        if in_fence:
            m = SHELL_CMD_RE.match(line)
            if m:
                commands.append(m.group(1).strip())

    return commands


def run_shell_commands(repo_root: Path, commands: List[str]) -> VerificationResult:
    if not commands:
        return VerificationResult(commands=[], passed=True, output="(no verification commands)\n")

    out_lines: List[str] = []
    for cmd in commands:
        effective_cmd = cmd
        stripped = cmd.strip()
        if stripped == "python" or stripped.startswith("python "):
            if shutil.which("python") is None and shutil.which("python3") is not None:
                effective_cmd = "python3" + stripped[len("python"):]
        out_lines.append(f"$ {effective_cmd}\n")
        p = subprocess.run(
            effective_cmd,
            shell=True,
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        out_lines.append(p.stdout)
        if p.returncode != 0:
            out_lines.append(f"[FAILED] exit code={p.returncode}\n")
            return VerificationResult(commands=commands, passed=False, output="".join(out_lines))

    return VerificationResult(commands=commands, passed=True, output="".join(out_lines))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_exists(repo_root: Path, rel: Path) -> Path:
    p = repo_root / rel
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {rel}")
    return p




def validate_output_schema(schema_path: Path) -> None:
    # Codex output schema is strict: for any object, 'required' must include every key in 'properties'.
    obj = load_json(schema_path)

    def check(node: Any, where: str) -> None:
        if isinstance(node, dict):
            if node.get('type') == 'object':
                props = node.get('properties') or {}
                req = node.get('required')
                if not isinstance(req, list):
                    raise ValueError(f"{where}: missing/invalid required list")
                missing = [k for k in props.keys() if k not in req]
                if missing:
                    raise ValueError(f"{where}: required missing keys: {missing}")
                for k, v in props.items():
                    check(v, where + f".properties.{k}")
            if 'items' in node:
                check(node['items'], where + '.items')

    check(obj, '$')


def _ensure_codex_auth(codex_home: Path, env: Dict[str, str]) -> None:
    auth_env_keys = (
        "OPENAI_API_KEY",
        "OPENAI_ACCESS_TOKEN",
        "OPENAI_API_TOKEN",
        "CODEX_API_KEY",
    )
    if any(env.get(k) for k in auth_env_keys):
        return

    auth_path = codex_home / "auth.json"
    config_path = codex_home / "config.toml"
    home_codex = Path.home() / ".codex"
    home_auth = home_codex / "auth.json"
    home_config = home_codex / "config.toml"

    try:
        if home_auth.exists() and not auth_path.exists():
            shutil.copy2(home_auth, auth_path)
            os.chmod(auth_path, 0o600)
        if home_config.exists() and not config_path.exists():
            shutil.copy2(home_config, config_path)
            os.chmod(config_path, 0o600)
    except Exception:
        # Best effort: avoid failing the loop if auth sync is not possible.
        return
def build_prompt(repo_root: Path, task_path: Path) -> str:
    wrapper = read_text(ensure_exists(repo_root, WRAPPER_PATH))
    task_md = read_text(task_path)
    return wrapper + "\n\n---\n\n" + task_md + "\n"


def run_codex(
    repo_root: Path,
    codex_cmd: str,
    prompt: str,
    log_dir: Path,
    full_auto: bool,
    yolo: bool,
) -> Tuple[int, str, str, Optional[Dict[str, Any]]]:
    """Run codex exec and return (exit_code, stdout, stderr, parsed_last_message_json)."""

    log_dir.mkdir(parents=True, exist_ok=True)
    last_msg_path = log_dir / "codex_last_message.json"

    cmd = [codex_cmd, "exec", "--color", "never", "--cd", str(repo_root)]

    # Safety presets
    if yolo:
        cmd.append("--yolo")
    elif full_auto:
        cmd.append("--full-auto")

    # Force JSON-shaped final response
    cmd += ["--output-schema", str(repo_root / SCHEMA_PATH), "--output-last-message", str(last_msg_path), "-"]

    env = os.environ.copy()
    if "CODEX_HOME" in env:
        codex_home = Path(env["CODEX_HOME"]).expanduser()
    else:
        codex_home = repo_root / ".codex"
        # Keep Codex session files inside the repo to avoid home-dir permission issues.
        env["CODEX_HOME"] = str(codex_home)
    codex_home.mkdir(parents=True, exist_ok=True)
    _ensure_codex_auth(codex_home, env)

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return (127, "", f"codex command not found: {codex_cmd}\n", None)

    stdout = proc.stdout
    stderr = proc.stderr

    # Save raw outputs
    (log_dir / "codex_stdout.txt").write_text(stdout, encoding="utf-8")
    (log_dir / "codex_stderr.txt").write_text(stderr, encoding="utf-8")

    parsed: Optional[Dict[str, Any]] = None
    if last_msg_path.exists():
        try:
            parsed = json.loads(last_msg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parsed = None

    return (proc.returncode, stdout, stderr, parsed)


def write_status_md(repo_root: Path, queue: List[Dict[str, Any]]) -> None:
    total = len(queue)
    done = sum(1 for t in queue if t.get("status") == "done")
    todo = sum(1 for t in queue if t.get("status") == "todo")
    doing = sum(1 for t in queue if t.get("status") == "doing")
    blocked = sum(1 for t in queue if t.get("status") == "blocked")

    next_task = pick_next_task(queue)
    next_line = f"{next_task['id']}: {next_task['title']}" if next_task else "(none)"

    lines = []
    lines.append(f"# STATUS\n\n")
    lines.append(f"- updated_at: {utc_now_iso()}\n")
    lines.append(f"- total: {total}, done: {done}, todo: {todo}, doing: {doing}, blocked: {blocked}\n")
    lines.append(f"- next: {next_line}\n\n")

    lines.append("## Queue\n\n")
    lines.append("| id | priority | status | title | depends_on |\n")
    lines.append("|---|---:|---|---|---|\n")
    for t in sorted(queue, key=lambda x: (int(x.get("priority", 999)), x.get("id", ""))):
        deps = ",".join(t.get("depends_on", []) or [])
        lines.append(
            f"| {t.get('id')} | {t.get('priority')} | {t.get('status')} | {t.get('title')} | {deps} |\n"
        )

    (repo_root / STATUS_MD).write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=None, help="repo root (default: auto-detect)")
    ap.add_argument("--codex-cmd", type=str, default="codex", help="codex CLI command name or path")
    ap.add_argument("--once", action="store_true", help="run only one runnable task")
    ap.add_argument("--task-id", type=str, default=None, help="run only the specified task id")
    ap.add_argument("--max-attempts", type=int, default=5, help="max retry attempts per task")
    ap.add_argument("--no-full-auto", action="store_true", help="do not pass --full-auto to codex")
    ap.add_argument("--yolo", action="store_true", help="pass --yolo (dangerous; use only in isolated env)")
    ap.add_argument("--list", action="store_true", help="list queue and exit")

    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path.cwd())
    # Validate Codex output schema early to avoid opaque 400 errors from the API
    schema_file = repo_root / SCHEMA_PATH
    try:
        validate_output_schema(schema_file)
    except Exception as e:
        print(f"ERROR: invalid Codex output schema at {schema_file}: {e}", file=sys.stderr)
        print("HINT: Ensure tools/codex_loop/response_schema.json includes verification.required=['commands','passed','notes']", file=sys.stderr)
        return 2

    queue = load_queue(repo_root)
    state = load_state(repo_root)

    if args.list:
        for t in sorted(queue, key=lambda x: (int(x.get("priority", 999)), x.get("id", ""))):
            deps = ",".join(t.get("depends_on", []) or [])
            print(f"{t['id']}\t{t.get('status')}\tP{t.get('priority')}\tdeps=[{deps}]\t{t.get('title')}")
        return 0

    LOGS_DIR_ABS = repo_root / LOGS_DIR
    LOGS_DIR_ABS.mkdir(parents=True, exist_ok=True)

    def get_task_by_id(tid: str) -> Optional[Dict[str, Any]]:
        for t in queue:
            if t.get("id") == tid:
                return t
        return None

    while True:
        task = None
        if args.task_id:
            task = get_task_by_id(args.task_id)
            if not task:
                print(f"Task not found: {args.task_id}", file=sys.stderr)
                return 2
            # respect dependencies
            if task.get("status") != "todo":
                print(f"Task {args.task_id} is not todo (status={task.get('status')}).", file=sys.stderr)
                return 2
            if not deps_done(task, tasks_by_id(queue)):
                print(f"Task {args.task_id} dependencies not satisfied.", file=sys.stderr)
                return 2
        else:
            task = pick_next_task(queue)
            if not task:
                print("No runnable todo tasks (all done, blocked, or waiting for deps).")
                write_status_md(repo_root, queue)
                save_json(repo_root / STATE_PATH, state)
                return 0

        task_id = task["id"]
        task_path = repo_root / Path(task["path"])
        if not task_path.exists():
            print(f"Task file missing: {task_path}", file=sys.stderr)
            task["status"] = "blocked"
            save_queue(repo_root, queue)
            write_status_md(repo_root, queue)
            return 2

        # attempts
        task_state = state.setdefault("tasks", {}).setdefault(task_id, {"attempts": 0})
        task_state["attempts"] = int(task_state.get("attempts", 0)) + 1
        task_state["last_started_at"] = utc_now_iso()

        if task_state["attempts"] > int(args.max_attempts):
            task["status"] = "blocked"
            task_state["last_error"] = f"Exceeded max attempts ({args.max_attempts})"
            save_queue(repo_root, queue)
            save_json(repo_root / STATE_PATH, state)
            write_status_md(repo_root, queue)
            return 2

        # mark doing
        task["status"] = "doing"
        save_queue(repo_root, queue)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_dir = LOGS_DIR_ABS / f"{task_id}_{ts}"
        log_dir.mkdir(parents=True, exist_ok=True)

        prompt = build_prompt(repo_root, task_path)
        (log_dir / "prompt.md").write_text(prompt, encoding="utf-8")

        rc, stdout, stderr, last_json = run_codex(
            repo_root=repo_root,
            codex_cmd=args.codex_cmd,
            prompt=prompt,
            log_dir=log_dir,
            full_auto=not args.no_full_auto,
            yolo=args.yolo,
        )

        task_state["codex_exit_code"] = rc
        task_state["last_log_dir"] = str(log_dir.relative_to(repo_root))

        if rc != 0:
            # codex failed to run
            task["status"] = "blocked" if rc == 127 else "todo"
            task_state["last_error"] = f"codex exec failed: rc={rc}"
            save_queue(repo_root, queue)
            save_json(repo_root / STATE_PATH, state)
            write_status_md(repo_root, queue)
            if args.task_id or args.once:
                return 2
            if task["status"] == "blocked":
                return 2
            continue

        # Run verification commands from the task file
        task_md = read_text(task_path)
        verif_cmds = extract_verification_commands(task_md)
        verif = run_shell_commands(repo_root, verif_cmds)
        (log_dir / "verification.log").write_text(verif.output, encoding="utf-8")

        # Decide outcome
        codex_status = (last_json or {}).get("status") if isinstance(last_json, dict) else None
        passed = verif.passed and codex_status == "done"

        if passed:
            task["status"] = "done"
            task_state["last_finished_at"] = utc_now_iso()
            task_state["last_status"] = "done"
        else:
            # If codex explicitly says blocked, honor it.
            if codex_status == "blocked":
                task["status"] = "blocked"
                task_state["last_status"] = "blocked"
                task_state["last_error"] = (last_json or {}).get("summary", "blocked")
            else:
                task["status"] = "todo"
                task_state["last_status"] = "todo"
                task_state["last_error"] = "verification failed" if not verif.passed else "codex status not done"

        save_queue(repo_root, queue)
        save_json(repo_root / STATE_PATH, state)
        write_status_md(repo_root, queue)

        if args.task_id or args.once:
            return 0 if passed else 2

        if task["status"] == "blocked":
            return 2

        # otherwise continue to next task


if __name__ == "__main__":
    raise SystemExit(main())
