import os
from collections import deque
from collections.abc import Iterable, Sequence

import pyarrow as pa
from dora import Node

ACTION_SIZE = 6


class AsyncActionExecutor:
    """Queue an action chunk and release one action per tick.

    A new chunk replaces actions that have not been executed yet. Actions
    whose timestep has already passed are dropped. The first
    ``blend_steps`` actions of a replacement chunk are cross-faded from
    the actions they replace, so the command does not jump at the seam.
    """

    def __init__(self, action_size: int = ACTION_SIZE, blend_steps: int = 10) -> None:
        self.action_size = action_size
        self.blend_steps = blend_steps
        self._pending: deque[list[float]] = deque()
        self.next_timestep = 0
        self.enabled = False

    def __len__(self) -> int:
        return len(self._pending)

    def start(self) -> None:
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False
        self._pending.clear()

    def push_chunk(
        self, chunk: Iterable[Sequence[float]], start_timestep: int | None = None
    ) -> int:
        actions = [[float(value) for value in action] for action in chunk]
        if any(len(action) != self.action_size for action in actions):
            raise ValueError(f"each action must have {self.action_size} values")
        start = self.next_timestep if start_timestep is None else start_timestep
        first = max(start, self.next_timestep)
        aligned = actions[first - start :]
        if self._pending and self.blend_steps > 0:
            old = list(self._pending)
            merged = []
            for index, action in enumerate(aligned):
                if index < self.blend_steps and index < len(old):
                    alpha = (index + 1) / (self.blend_steps + 1)
                    merged.append(
                        [
                            (1.0 - alpha) * before + alpha * after
                            for before, after in zip(old[index], action)
                        ]
                    )
                else:
                    merged.append(action)
            aligned = merged
        self._pending = deque(aligned)
        self.next_timestep = first
        return len(self._pending)

    def step(self) -> list[float] | None:
        if not self.enabled or not self._pending:
            return None
        action = self._pending.popleft()
        self.next_timestep += 1
        return action


def _chunk(value) -> list[list[float]]:
    rows = value.to_pylist()
    if not rows:
        return []
    if isinstance(rows[0], (list, tuple)):
        return rows
    if len(rows) % ACTION_SIZE:
        raise ValueError(f"flat action chunk length must be a multiple of {ACTION_SIZE}")
    return [rows[index : index + ACTION_SIZE] for index in range(0, len(rows), ACTION_SIZE)]


def _start_timestep(event) -> int | None:
    raw = event["metadata"].get("timestep")
    return None if raw is None else int(raw)


def _command_value(event) -> str | None:
    value = event["value"]
    try:
        if hasattr(value, "as_py"):
            return value[0].as_py()
        return value.to_pylist()[0]
    except Exception:
        return None


def main() -> None:
    node = Node()
    executor = AsyncActionExecutor(blend_steps=int(os.environ.get("BLEND_STEPS", "10")))

    def _send_status() -> None:
        node.send_output(
            "queue_status",
            pa.array([len(executor), executor.next_timestep], type=pa.int64()),
        )

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "command":
            command = _command_value(event)
            if command == "start":
                executor.start()
                print("executor started", flush=True)
            elif command == "stop":
                executor.stop()
                print("executor stopped; queue cleared", flush=True)
            else:
                print(f"unknown command {command!r}; skipping", flush=True)
                continue
            _send_status()
            continue
        if event["id"] == "action_chunk":
            try:
                queued = executor.push_chunk(_chunk(event["value"]), _start_timestep(event))
            except Exception as exc:
                print(f"action chunk rejected ({exc}); skipping", flush=True)
                continue
            print(f"action chunk queued ({queued})", flush=True)
            _send_status()
            continue
        if event["id"] == "tick":
            action = executor.step()
            if action is not None:
                node.send_output("action", pa.array(action, type=pa.float32()))
                _send_status()


if __name__ == "__main__":
    main()
