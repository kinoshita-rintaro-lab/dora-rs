import os
import time

import pyarrow as pa
from dora import Node
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

def _publish_state(follower: SO101Follower, node: Node) -> None:
    state = follower.get_observation()
    measured = [state[f"{name}.pos"] for name in JOINTS]
    node.send_output("follower_state", pa.array(measured, type=pa.float32()))


def _apply(follower: SO101Follower, node: Node, angle: list[float]) -> None:
    follower.send_action({f"{name}.pos": value for name, value in zip(JOINTS, angle)})
    _publish_state(follower, node)


def _command_value(event) -> str | None:
    value = event["value"]
    try:
        if hasattr(value, "as_py"):
            return value[0].as_py()
        return value.to_pylist()[0]
    except Exception:
        return None


def _release(follower: SO101Follower, *, disable_torque: bool) -> None:
    bus = follower.bus
    if not bus.is_connected:
        return
    try:
        bus.disconnect(disable_torque=disable_torque)
    except ConnectionError as exc:
        print(f"follower disconnect failed ({exc})", flush=True)
        if bus.is_connected:
            bus.port_handler.closePort()


def main() -> None:
    follower = SO101Follower(SO101FollowerConfig(port=os.environ["PORT"], id="so101-follower"))
    node = Node()
    connected = False
    torque_enabled = False
    retry_at = 0.0

    def _set_torque(enabled: bool) -> None:
        nonlocal torque_enabled
        if not connected or torque_enabled == enabled:
            return
        if enabled:
            follower.bus.enable_torque()
        else:
            follower.bus.disable_torque()
        torque_enabled = enabled
        try:
            readback = follower.bus.sync_read("Torque_Enable")
        except Exception as exc:
            readback = f"readback failed ({exc})"
        print(
            f"follower torque {'on' if enabled else 'off'} (readback: {readback})",
            flush=True,
        )

    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] not in ("tick", "cmd", "command"):
                continue
            if not connected:
                now = time.monotonic()
                if now < retry_at:
                    continue
                try:
                    follower.connect(calibrate=False)
                except ConnectionError as exc:
                    print(f"follower connect failed ({exc}); retrying", flush=True)
                    _release(follower, disable_torque=False)
                    retry_at = now + 1.0
                    continue
                connected = True
                # connect() leaves torque on; the arm must stay limp
                # until inference is started from the UI.
                torque_enabled = True
                try:
                    _set_torque(False)
                except ConnectionError as exc:
                    print(f"follower torque off failed ({exc})", flush=True)

            try:
                if event["id"] == "tick":
                    _publish_state(follower, node)
                elif event["id"] == "cmd":
                    _apply(follower, node, event["value"].to_pylist())
                else:
                    command = _command_value(event)
                    if command == "start":
                        _set_torque(True)
                    elif command == "stop":
                        _set_torque(False)
                    else:
                        print(f"unknown command {command!r}; skipping", flush=True)
            except ConnectionError as exc:
                print(f"follower link lost ({exc}); skipping", flush=True)
    finally:
        _release(follower, disable_torque=True)


if __name__ == "__main__":
    main()