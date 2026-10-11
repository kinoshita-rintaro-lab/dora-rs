import os

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

def _apply(follower: SO101Follower, node: Node, angle: list[float]) -> None:
    follower.send_action({f"{name}.pos": value for name, value in zip(JOINTS, angle)})
    state = follower.get_observation()
    measured = [state[f"{name}.pos"] for name in JOINTS]
    node.send_output("follower_state", pa.array(measured, type=pa.float32()))


def _reopen(follower: SO101Follower) -> None:
    """Reopen the serial port without disabling torque, so the arm keeps holding."""
    bus = follower.bus
    if bus.is_connected:
        try:
            bus.disconnect(disable_torque=False)
        except ConnectionError:
            bus.port_handler.closePort()
    bus.connect()


def main() -> None:
    follower = SO101Follower(SO101FollowerConfig(port=os.environ["PORT"], id="so101-follower"))
    follower.connect(calibrate=False)
    node = Node()
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] != "cmd":  # skip when not reading or not cmd
                continue

            angle = event["value"].to_pylist()
            try:
                _apply(follower, node, angle)
            except ConnectionError as exc:
                print(f"follower link lost ({exc}); reopening", flush=True)
                try:
                    _reopen(follower)
                    _apply(follower, node, angle)
                except ConnectionError as retry_exc:
                    print(f"follower still down ({retry_exc})", flush=True)
    finally:
        try:
            follower.disconnect()
        except ConnectionError as exc:
            print(f"follower disconnect failed ({exc})", flush=True)
            if follower.bus.is_connected:
                follower.bus.port_handler.closePort()
if __name__ == "__main__":
    main()