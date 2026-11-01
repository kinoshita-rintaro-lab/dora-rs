"""Web UI node for remote SO-101 inference.

Serves a small control page over HTTP and bridges it to Dora: the page
shows the `status` reported by dora-so101-policy-client, and Start /
Stop presses become `command` outputs. Start carries the task text in
its metadata. The same presses also emit a `torque` bool for the
follower: false until Start, true while running, false again on Stop.
Nothing is inferred or executed until Start is pressed.
"""

import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
from dora import Node

ALLOWED_COMMANDS = {"start", "stop"}

_PAGE = Path(__file__).with_name("page.html").read_text(encoding="utf-8")


class UiState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status: dict | None = None
        self.commands: queue.Queue = queue.Queue()


def _make_server(state: UiState, host: str, port: int, task_default: str):
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
            elif self.path == "/api/status":
                with state.lock:
                    status = state.status
                self._json(200, {"status": status, "task_default": task_default})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/api/command":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "invalid JSON body"})
                return
            command = body.get("command")
            if command not in ALLOWED_COMMANDS:
                self._json(400, {"error": f"unknown command {command!r}"})
                return
            item = {"command": command}
            if command == "start":
                item["task"] = str(body.get("task") or "")
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

    state = UiState()
    server = _make_server(state, host, port, task_default)
    threading.Thread(target=server.serve_forever, daemon=True, name="inference-ui-http").start()
    print(f"inference UI at http://{host}:{port}", flush=True)

    node = Node()
    torque_on = False
    torque_sent = False

    def _send_torque(enabled: bool) -> None:
        nonlocal torque_on, torque_sent
        if torque_sent and torque_on == enabled:
            return
        node.send_output("torque", pa.array([enabled], type=pa.bool_()))
        torque_on = enabled
        torque_sent = True

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
                # Released until the operator presses Start.
                if not torque_sent:
                    _send_torque(False)
                while True:
                    try:
                        item = state.commands.get_nowait()
                    except queue.Empty:
                        break
                    _send_torque(item["command"] == "start")
                    metadata = {}
                    if item["command"] == "start" and item.get("task"):
                        metadata["task"] = item["task"]
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
