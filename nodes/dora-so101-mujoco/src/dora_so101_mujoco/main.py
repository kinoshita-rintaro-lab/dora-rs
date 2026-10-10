import math
from pathlib import Path

import mujoco
import mujoco.viewer
from dora import Node

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_CLOSED = -0.17453
GRIPPER_OPEN = 1.74533
SCENE = Path(__file__).parent / "so101" / "scene.xml"

def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    node = Node()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] != "leader_cmd":
                continue
            values = event["value"].to_pylist()
            if len(values) != len(JOINTS):
                raise RuntimeError(f"leader_cmd length {len(values)}")
            for name, value in zip(JOINTS, values):
                if name == "gripper":
                    target = GRIPPER_CLOSED + (value / 100.0) * (GRIPPER_OPEN - GRIPPER_CLOSED)
                elif name == "wrist_roll":
                    target = math.radians(value+90)
                else:
                    target = math.radians(value)
                data.ctrl[model.actuator(name).id] = target
            steps = max(1, round(0.016 / model.opt.timestep))
            for _ in range(steps):
                mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()