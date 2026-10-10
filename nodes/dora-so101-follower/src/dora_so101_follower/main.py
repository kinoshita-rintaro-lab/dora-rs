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

def main() -> None:
    follower = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM0", id="so101-follower"))
    follower.connect(calibrate=False)
    node = Node()
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] != "cmd": # skip when not reading or not cmd
                continue
            
            angle = event["value"].to_pylist()
            follower.send_action({f"{name}.pos": angle for name, angle in zip(JOINTS, angle)})
            
            state = follower.get_observation()
            measured = [state[f"{name}.pos"] for name in JOINTS]
            node.send_output("follower_state", pa.array(measured, type=pa.float32()))
    finally:
        follower.disconnect()
if __name__ == "__main__":
    main()