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
GRIPPER = JOINTS.index("gripper")


def _gripper() -> dict:
    mode = os.environ.get("GRIPPER_MODE", "discrete").strip().lower()
    if mode not in ("discrete", "continuous"):
        print(f"unknown GRIPPER_MODE {mode!r}; using discrete", flush=True)
        mode = "discrete"
    gripper = {
        "discrete": mode == "discrete",
        "threshold": float(os.environ.get("GRIPPER_THRESHOLD", "10")),
        "closed": float(os.environ.get("GRIPPER_CLOSED", "0")),
        "open": float(os.environ.get("GRIPPER_OPEN", "30")),
    }
    if gripper["discrete"]:
        print(
            "follower gripper discrete "
            f"(threshold {gripper['threshold']}, closed {gripper['closed']}, open {gripper['open']})",
            flush=True,
        )
    else:
        print("follower gripper continuous", flush=True)
    return gripper


def _publish_state(follower: SO101Follower, node: Node, gripper: dict) -> None:
    state = follower.get_observation()
    measured = [state[f"{name}.pos"] for name in JOINTS]
    if gripper["discrete"]:
        measured[GRIPPER] = 1.0 if measured[GRIPPER] >= gripper["threshold"] else 0.0
    node.send_output("follower_state", pa.array(measured, type=pa.float32()))


def _apply(follower: SO101Follower, node: Node, angle: list[float], gripper: dict) -> None:
    if gripper["discrete"]:
        angle = list(angle)
        angle[GRIPPER] = gripper["open"] if float(angle[GRIPPER]) >= 0.5 else gripper["closed"]
    follower.send_action({f"{name}.pos": value for name, value in zip(JOINTS, angle)})
    _publish_state(follower, node, gripper)


def _torque_value(event) -> bool | None:
    """Shared torque signal. True holds position, false releases the arm."""
    value = event["value"]
    try:
        raw = value[0].as_py() if hasattr(value, "as_py") else value.to_pylist()[0]
    except Exception:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1, 0.0, 1.0):
        return bool(raw)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("1", "on", "true"):
            return True
        if text in ("0", "off", "false"):
            return False
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
    gripper = _gripper()
    follower = SO101Follower(SO101FollowerConfig(port=os.environ["PORT"], id="so101-follower"))
    node = Node()
    connected = False
    # connect() enables torque. Graphs that never send `torque` stay that way
    # and keep following cmd. A later false releases the arm.
    torque_enabled = True
    torque_request: bool | None = None
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
        print(f"follower torque {'on' if enabled else 'off'}", flush=True)

    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] not in ("tick", "cmd", "torque"):
                continue
            if event["id"] == "torque":
                requested = _torque_value(event)
                if requested is None:
                    print("torque signal rejected; skipping", flush=True)
                else:
                    torque_request = requested
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
                torque_enabled = True
                if torque_request is not None:
                    try:
                        _set_torque(torque_request)
                    except ConnectionError as exc:
                        print(f"follower torque update failed ({exc})", flush=True)

            try:
                if event["id"] == "torque":
                    if torque_request is not None:
                        _set_torque(torque_request)
                elif event["id"] == "tick":
                    _publish_state(follower, node, gripper)
                else:
                    # A goal written after torque-off turns the servo back into
                    # position hold. Drop anything still queued.
                    if not torque_enabled:
                        continue
                    _apply(follower, node, event["value"].to_pylist(), gripper)
            except ConnectionError as exc:
                print(f"follower link lost ({exc}); skipping", flush=True)
    finally:
        _release(follower, disable_torque=True)


if __name__ == "__main__":
    main()
