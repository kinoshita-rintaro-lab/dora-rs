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
    retry_at = 0.0
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] not in ("tick", "cmd"):
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

            try:
                if event["id"] == "tick":
                    _publish_state(follower, node)
                else:
                    _apply(follower, node, event["value"].to_pylist())
            except ConnectionError as exc:
                print(f"follower link lost ({exc}); skipping", flush=True)
    finally:
        _release(follower, disable_torque=True)


if __name__ == "__main__":
    main()