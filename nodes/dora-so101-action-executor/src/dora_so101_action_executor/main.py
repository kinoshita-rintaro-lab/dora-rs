from collections import deque
from collections.abc import Iterable, Sequence

import pyarrow as pa
from dora import Node

ACTION_SIZE = 6


class AsyncActionExecutor:
    """Queue an action chunk and release one action per tick.

    A new chunk replaces actions that have not been executed yet. Actions
    whose timestep has already passed are dropped.
    """

    def __init__(self, action_size: int = ACTION_SIZE) -> None:
        self.action_size = action_size
        self._pending: deque[list[float]] = deque()
        self.next_timestep = 0

    def __len__(self) -> int:
        return len(self._pending)

    def push_chunk(
        self, chunk: Iterable[Sequence[float]], start_timestep: int | None = None
    ) -> int:
        actions = [[float(value) for value in action] for action in chunk]
        if any(len(action) != self.action_size for action in actions):
            raise ValueError(f"each action must have {self.action_size} values")
        start = self.next_timestep if start_timestep is None else start_timestep
        first = max(start, self.next_timestep)
        self._pending = deque(actions[first - start :])
        self.next_timestep = first
        return len(self._pending)

    def step(self) -> list[float] | None:
        if not self._pending:
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


def main() -> None:
    node = Node()
    executor = AsyncActionExecutor()

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
