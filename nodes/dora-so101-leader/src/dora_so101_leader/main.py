import os

import pyarrow as pa
from dora import Node
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

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
    }
    if gripper["discrete"]:
        print(f"leader gripper discrete (threshold {gripper['threshold']})", flush=True)
    else:
        print("leader gripper continuous", flush=True)
    return gripper


def main() -> None:
    gripper = _gripper()
    leader = SO101Leader(SO101LeaderConfig(port=os.environ["PORT"], id="so101-leader"))
    leader.connect(calibrate=False)
    node = Node()
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] != "tick": # skip when not reading or not tick
                continue
            action = leader.get_action()
            missing = [name for name in JOINTS if f"{name}.pos" not in action]
            if missing:
                raise RuntimeError(f"motor read missing: {missing}")
            command = [action[f"{name}.pos"] for name in JOINTS]
            if gripper["discrete"]:
                command[GRIPPER] = 1.0 if command[GRIPPER] >= gripper["threshold"] else 0.0
            node.send_output("leader_cmd", pa.array(command, type=pa.float32()))
    finally:
        leader.disconnect()
if __name__ == "__main__":
    main()