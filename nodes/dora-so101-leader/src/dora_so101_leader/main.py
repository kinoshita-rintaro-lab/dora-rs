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

def main() -> None:
    leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM1", id="so101-leader"))
    leader.connect(calibrate=False)
    node = Node()
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] != "tick": # skip when not reading or not tick
                continue
            action = leader.get_action()
            command = pa.array([action[f"{name}.pos"] for name in JOINTS], type=pa.float32())
            missing = [name for name in JOINTS if f"{name}.pos" not in action]
            if missing:
                raise RuntimeError(f"motor read missing: {missing}")
            node.send_output("leader_cmd", command)
    finally:
        leader.disconnect()
if __name__ == "__main__":
    main()