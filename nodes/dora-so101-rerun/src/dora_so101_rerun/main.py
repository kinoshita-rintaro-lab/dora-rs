import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from dora import Node
from lerobot.utils.rerun_visualization import init_rerun, log_rerun_data, shutdown_rerun

_ENCODED = {"jpeg", "jpg", "jpe", "png", "bmp", "webp"}


def _ensure_viewer_on_path() -> None:
    """Make the Rerun Viewer executable findable for ``rr.spawn``.

    The viewer ships in the virtual environment's bin directory, but a
    spawned node's PATH does not necessarily contain it.
    """
    if shutil.which("rerun"):
        return
    bin_dir = Path(sys.prefix) / "bin"
    if (bin_dir / "rerun").exists():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _items(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _image(event: dict) -> np.ndarray | None:
    storage = event["value"].to_numpy()
    encoding = str(event["metadata"].get("encoding", "jpeg")).lower()
    if encoding in _ENCODED:
        frame = cv2.imdecode(storage, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height = int(event["metadata"]["height"])
    width = int(event["metadata"]["width"])
    frame = storage.reshape((height, width, 3))
    if encoding == "bgr8":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def _values(event: dict, source: str, joints: list[str]) -> dict[str, float | np.ndarray]:
    values = event["value"].to_numpy().reshape(-1)
    if not joints:
        return {source: values}
    return {
        f"{source}.{name}": float(value)
        for name, value in zip(joints, values, strict=True)
    }


def _series(prefix: str, sources: list[str], joints: list[str]) -> list[str]:
    if joints:
        return [f"{prefix}.{source}.{joint}" for source in sources for joint in joints]
    return [f"{prefix}.{source}" for source in sources]


def _stack(views: list) -> rrb.ContainerLike | rrb.View | None:
    if not views:
        return None
    if len(views) == 1:
        return views[0]
    return rrb.Vertical(*views)


def _overview(images: list[str], observations: list[str], actions: list[str], joints: list[str]):
    cameras = [rrb.Spatial2DView(origin=f"observation.{name}", name=name) for name in images]
    plots = []
    if observations:
        plots.append(
            rrb.TimeSeriesView(name="observation", contents=_series("observation", observations, joints))
        )
    if actions:
        plots.append(rrb.TimeSeriesView(name="action", contents=_series("action", actions, joints)))
    camera_column = _stack(cameras)
    plot_column = _stack(plots)
    if camera_column is not None and plot_column is not None:
        return rrb.Horizontal(camera_column, plot_column, column_shares=[2, 3], name="overview")
    return camera_column or plot_column


def _joints(observations: list[str], actions: list[str], joints: list[str]):
    views = []
    for joint in joints:
        contents = [f"action.{source}.{joint}" for source in actions]
        contents += [f"observation.{source}.{joint}" for source in observations]
        if contents:
            views.append(rrb.TimeSeriesView(name=joint, contents=contents))
    return _stack(views)


def _layout(images: list[str], observations: list[str], actions: list[str], joints: list[str]) -> None:
    overview = _overview(images, observations, actions, joints)
    per_joint = _joints(observations, actions, joints) if joints else None
    if overview is not None and per_joint is not None:
        root = rrb.Tabs(overview, per_joint, active_tab=0)
    else:
        root = overview or per_joint
    if root is None:
        return
    rr.send_blueprint(rrb.Blueprint(root))
    # log_rerun_data freezes its layout on the first call. Keep this one.
    log_rerun_data.blueprint = True


def main() -> None:
    images = _items("IMAGES")
    actions = _items("ACTIONS")
    observations = _items("OBSERVATIONS")
    joints = _items("JOINTS")
    image_ids = set(images)
    action_ids = set(actions)
    observation_ids = set(observations)

    _ensure_viewer_on_path()
    init_rerun(os.environ.get("SESSION", "so101"))
    _layout(images, observations, actions, joints)
    node = Node()
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            source = event["id"]
            if source in image_ids:
                frame = _image(event)
                if frame is not None:
                    log_rerun_data(observation={source: frame})
            elif source in action_ids:
                log_rerun_data(action=_values(event, source, joints))
            elif source in observation_ids:
                log_rerun_data(observation=_values(event, source, joints))
    finally:
        shutdown_rerun()


if __name__ == "__main__":
    main()
