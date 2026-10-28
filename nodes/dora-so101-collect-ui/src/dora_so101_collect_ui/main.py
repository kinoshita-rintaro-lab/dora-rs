"""Web UI node for the SO-101 episode recorder.

Serves a small control page over HTTP and bridges it to Dora: button
presses become `command` outputs for dora-so101-record, and the
recorder's `status` output drives what the page shows. The page never
guesses the recording state on its own.
"""

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
from dora import Node

ALLOWED_COMMANDS = {
    "start",
    "end",
    "success",
    "fail",
    "cancel",
    "mark_success",
    "mark_fail",
}

_PAGE = Path(__file__).with_name("page.html").read_text(encoding="utf-8")
_LIBRARY_LOCK = threading.Lock()


class UiState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status: dict | None = None
        self.commands: queue.Queue = queue.Queue()


def _library_path() -> Path:
    raw = os.environ.get("TASK_LIBRARY")
    if raw:
        return Path(raw).expanduser()
    for base in (Path.cwd(), Path.cwd().parent):
        if (base / "record").is_dir():
            return base / "record" / "task-library.json"
    return Path.cwd() / "task-library.json"


def _clean_library(payload: dict) -> dict:
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups or len(groups) > 100:
        raise ValueError("invalid groups")
    cleaned = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("invalid group")
        group_id = str(group.get("id") or "").strip()
        if not group_id:
            raise ValueError("group id required")
        raw_tasks = group.get("tasks", [])
        if not isinstance(raw_tasks, list) or len(raw_tasks) > 500:
            raise ValueError("invalid tasks")
        tasks = []
        for task in raw_tasks:
            text = str(task).strip()
            if text:
                tasks.append(text[:2000])
        cleaned.append({"id": group_id[:80], "name": str(group.get("name") or "").strip()[:80], "tasks": tasks})
    active = str(payload.get("active") or cleaned[0]["id"])
    if active not in {group["id"] for group in cleaned}:
        active = cleaned[0]["id"]
    return {"active": active, "groups": cleaned}


def _read_library(path: Path) -> dict | None:
    with _LIBRARY_LOCK:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def _write_library(path: Path, payload: dict) -> dict:
    library = _clean_library(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    text = json.dumps(library, ensure_ascii=False, indent=2) + "\n"
    with _LIBRARY_LOCK:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return library


def _make_server(state: UiState, host: str, port: int, task_default: str, result_prompt_default: bool, library_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep node output readable
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                body = _PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.split("?", 1)[0] == "/api/library":
                stored = _read_library(library_path)
                self._json(200, {"stored": stored is not None, **(stored or {"active": "", "groups": []})})
            elif self.path == "/api/status":
                with state.lock:
                    status = state.status
                self._json(
                    200,
                    {
                        "status": status,
                        "task_default": task_default,
                        "result_prompt_default": result_prompt_default,
                    },
                )
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path not in ("/api/command", "/api/library"):
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "invalid JSON body"})
                return
            if path == "/api/library":
                try:
                    library = _write_library(library_path, body)
                except (ValueError, OSError) as exc:
                    self._json(400, {"error": str(exc)})
                    return
                self._json(200, {"stored": True, **library})
                return
            command = body.get("command")
            if command not in ALLOWED_COMMANDS:
                self._json(400, {"error": f"unknown command {command!r}"})
                return
            item = {"command": command}
            if command == "start":
                item["task"] = str(body.get("task") or "")
                try:
                    item["task_index"] = int(body.get("task_index", 0))
                except (TypeError, ValueError):
                    item["task_index"] = 0
            state.commands.put(item)
            self._json(202, {"queued": command})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def _status_value(event: dict) -> dict:
    value = event["value"]
    if hasattr(value, "as_py"):
        return json.loads(value[0].as_py())
    return json.loads(value.to_pylist()[0])


def main() -> None:
    host = os.environ.get("UI_HOST", "127.0.0.1")
    port = int(os.environ.get("UI_PORT", "8080"))
    task_default = os.environ.get("TASK", "")
    result_prompt_default = os.environ.get("RESULT_PROMPT", "on").lower() not in ("0", "false", "off", "no")

    state = UiState()
    server = _make_server(state, host, port, task_default, result_prompt_default, _library_path())
    threading.Thread(target=server.serve_forever, daemon=True, name="record-ui-http").start()
    print(f"recorder UI at http://{host}:{port}", flush=True)

    node = Node()
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            if event["id"] == "status":
                try:
                    status = _status_value(event)
                except Exception as exc:
                    print(f"status rejected ({exc}); skipping", flush=True)
                    continue
                with state.lock:
                    state.status = status
            elif event["id"] == "tick":
                while True:
                    try:
                        item = state.commands.get_nowait()
                    except queue.Empty:
                        break
                    metadata = {}
                    if item["command"] == "start":
                        if item.get("task"):
                            metadata["task"] = item["task"]
                        metadata["task_index"] = item.get("task_index", 0)
                    node.send_output(
                        "command",
                        pa.array([item["command"]]),
                        metadata if metadata else None,
                    )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
