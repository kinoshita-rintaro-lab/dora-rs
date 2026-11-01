import os
from collections import deque
from collections.abc import Iterable, Sequence

import pyarrow as pa
from dora import Node

ACTION_SIZE = 6


class AsyncActionExecutor:
    """Queue an action chunk and release one action per tick.

    A new chunk replaces actions that have not been executed yet. Actions
    whose timestep has already passed are dropped. Unless ``rtc`` is set,
    the replacement is shifted so its first action matches the action
    currently commanded (each inference is an independent sample and its
    absolute joint targets disagree), then the first ``blend_steps``
    actions are cross-faded from the actions they replace. With ``rtc``
    the model already continues the previous chunk, so the replacement
    is used as produced.
    """

    def __init__(
        self,
        action_size: int = ACTION_SIZE,
        blend_steps: int = 10,
        *,
        sync: bool = False,
        rtc: bool = False,
    ) -> None:
        self.action_size = action_size
        self.blend_steps = blend_steps
        self.sync = sync
        self.rtc = rtc
        self._pending: deque[list[float]] = deque()
        self._last_action: list[float] | None = None
        self._smooth: list[float] | None = None
        self.smooth_alpha = 1.0
        self.next_timestep = 0
        self.enabled = False

    def __len__(self) -> int:
        return len(self._pending)

    def start(self) -> None:
        self.enabled = True
        self._last_action = None
        self._smooth = None

    def stop(self) -> None:
        self.enabled = False
        self._pending.clear()
        self._last_action = None
        self._smooth = None

    def push_chunk(
        self, chunk: Iterable[Sequence[float]], start_timestep: int | None = None
    ) -> int:
        actions = [[float(value) for value in action] for action in chunk]
        if any(len(action) != self.action_size for action in actions):
            raise ValueError(f"each action must have {self.action_size} values")
        if self.sync and self._pending:
            return len(self._pending)
        start = self.next_timestep if start_timestep is None else start_timestep
        first = max(start, self.next_timestep)
        aligned = actions[first - start :]
        # RTC chunks already follow the previous plan. Shifting them again
        # would stack a second correction on the model's.
        align = not self.sync and not self.rtc
        anchor = (self._pending[0] if self._pending else self._last_action) if align else None
        if anchor is not None and aligned:
            delta = [anchor[i] - aligned[0][i] for i in range(self.action_size)]
            aligned = [[value + shift for value, shift in zip(action, delta)] for action in aligned]
        if align and self._pending and self.blend_steps > 0:
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
        # The queue and the async anchor stay on the model's action.
        # Only the value sent to the arm is low-passed.
        self._last_action = list(action)
        self.next_timestep += 1
        return self._smooth_action(action)

    def _smooth_action(self, action: list[float]) -> list[float]:
        """One-pole low-pass. The first action after start passes through."""
        if self.smooth_alpha >= 1.0 or self._smooth is None:
            self._smooth = list(action)
            return list(action)
        smoothed = [
            previous + self.smooth_alpha * (value - previous)
            for previous, value in zip(self._smooth, action)
        ]
        self._smooth = smoothed
        return smoothed


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
    mode = os.environ.get("INFERENCE_MODE", "async")
    if mode not in ("sync", "async", "rtc"):
        print(f"unknown INFERENCE_MODE {mode!r}; using async", flush=True)
        mode = "async"
    executor = AsyncActionExecutor(
        blend_steps=int(os.environ.get("BLEND_STEPS", "10")),
        sync=mode == "sync",
        rtc=mode == "rtc",
    )
    raw_alpha = os.environ.get("ACTION_SMOOTH", "0.35")
    try:
        alpha = float(raw_alpha)
    except ValueError:
        alpha = -1.0
    if not 0.0 < alpha <= 1.0:
        print(f"ACTION_SMOOTH {raw_alpha!r} is outside (0, 1]; using 0.35", flush=True)
        alpha = 0.35
    executor.smooth_alpha = alpha
    print(f"action executor inference mode: {mode}, smooth {alpha:g}", flush=True)

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
