"""Turn a dora-so101-record dataset into one LeRobot dataset.

Each episode is resampled on a fixed clock. A frame keeps the latest leader
command, follower measurement, and camera image that had already arrived.
The command becomes the LeRobot action and the follower measurement becomes
observation.state. The episode's own task text is stored on every frame.
Cameras are encoded to MP4. Older episodes that stored these streams as
action/ and obs/ are read the same way.

An episode that is missing a camera from the recording setup is an error.
Pass --skip-incomplete to leave those episodes out. Episodes whose success
flag is still unset are skipped. Re-running against an existing LeRobot
dataset appends another copy of the episodes.

    uv run tools/record2lerobot.py record/so101 --root datasets/so101
"""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _metadata(path: Path) -> dict:
    metadata_path = path / "metadata.yaml"
    if not metadata_path.is_file() or not (path / "episodes").is_dir():
        raise SystemExit(f"{path} is not a record dataset (expected metadata.yaml and episodes/)")
    with metadata_path.open(encoding="utf-8") as source:
        metadata = yaml.safe_load(source) or {}
    if not isinstance(metadata, dict) or not isinstance(metadata.get("episodes"), list):
        raise SystemExit(f"{metadata_path} has no episode list")
    return metadata


def _names(metadata: dict) -> tuple[list[str], list[str], str]:
    robot = metadata.get("robot") or {}
    joints = list(robot.get("joints") or JOINTS)
    cameras = list((metadata.get("frequencies") or {}).get("cameras") or [])
    robot_type = str(robot.get("type") or "so101")
    if not joints:
        raise SystemExit("metadata.yaml does not name any joints")
    return joints, cameras, robot_type


def _selected(metadata: dict, episode_ids: list[str] | None, success_only: bool) -> list[dict]:
    episodes = metadata["episodes"]
    known = {str(episode.get("id")) for episode in episodes}
    if episode_ids:
        missing = [episode_id for episode_id in episode_ids if episode_id not in known]
        if missing:
            raise SystemExit(f"unknown episode(s): {', '.join(missing)}")
    chosen = []
    for episode in episodes:
        episode_id = str(episode.get("id"))
        if episode_ids and episode_id not in episode_ids:
            continue
        if episode.get("success") is None:
            print(f"episode {episode_id}: result still pending; skipping")
            continue
        if success_only and episode.get("success") is not True:
            print(f"episode {episode_id}: not marked success; skipping")
            continue
        chosen.append(episode)
    if not chosen:
        raise SystemExit("no episodes to convert")
    return chosen


def _vectors(path: Path) -> tuple[list[int], list[np.ndarray]]:
    if not path.is_file():
        return [], []
    table = pq.read_table(path)
    times = table.column("timestamp").cast(pa.int64()).to_pylist()
    values = [np.asarray(item.as_py(), dtype=np.float32).reshape(-1) for item in table.column("value")]
    order = sorted(range(len(times)), key=times.__getitem__)
    return [times[index] for index in order], [values[index] for index in order]


def _camera_paths(directory: Path) -> tuple[list[int], list[Path]]:
    if not directory.is_dir():
        return [], []
    frames = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
            frames.append((int(path.stem), path))
    frames.sort()
    return [stamp for stamp, _ in frames], [path for _, path in frames]


def _decode(path: Path) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"could not decode {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _at(times: list[int], moment_ns: int) -> int:
    index = bisect.bisect_right(times, moment_ns) - 1
    if index < 0:
        raise SystemExit("a frame has no sample at or before its timestamp")
    return index


def _moments(streams: dict[str, list[int]], fps: int) -> list[int]:
    start = max(times[0] for times in streams.values())
    end = min(times[-1] for times in streams.values())
    if end < start:
        raise SystemExit("cmd, follower, and camera streams do not overlap")
    step = round(1_000_000_000 / fps)
    return list(range(start, end + 1, step))


def _features(joints: list[str], shapes: dict[str, tuple[int, int]]) -> dict:
    vector = {"dtype": "float32", "shape": (len(joints),), "names": list(joints)}
    features: dict = {"action": vector, "observation.state": vector}
    for name, (height, width) in shapes.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _shape(feature: dict) -> tuple[int, ...]:
    return tuple(int(item) for item in feature["shape"])


def _open_dataset(root: Path, repo_id: str, fps: int, robot_type: str, joints: list[str], shapes: dict[str, tuple[int, int]]) -> LeRobotDataset:
    info = root / "meta" / "info.json"
    if info.is_file():
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=root, streaming_encoding=True)
        if dataset.meta.fps != fps:
            raise SystemExit(f"{root} was recorded at {dataset.meta.fps} fps; this conversion uses {fps}")
        image_keys = [key for key, feature in dataset.meta.features.items() if feature["dtype"] == "image"]
        if image_keys:
            raise SystemExit(f"{root} stores {', '.join(image_keys)} as images; this conversion writes MP4 only")
        expected = {f"observation.images.{name}" for name in shapes}
        actual = {key for key, feature in dataset.meta.features.items() if feature["dtype"] == "video"}
        if actual != expected:
            raise SystemExit(f"{root} cameras are {sorted(actual)}; this recording has {sorted(expected)}")
        for name, shape in shapes.items():
            stored = _shape(dataset.meta.features[f"observation.images.{name}"])
            if stored != (shape[0], shape[1], 3):
                raise SystemExit(f"{root} {name} is {stored[1]}x{stored[0]}; this recording is {shape[1]}x{shape[0]}")
        action = dataset.meta.features.get("action")
        if action is None or _shape(action) != (len(joints),):
            raise SystemExit(f"{root} action width does not match {len(joints)} joints")
        return dataset
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} exists and is not a LeRobot dataset")
    if root.exists():
        root.rmdir()
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type=robot_type,
        use_videos=True,
        streaming_encoding=True,
        features=_features(joints, shapes),
    )


def _stream_file(episode_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = episode_dir / name / "state.parquet"
        if path.is_file():
            return path
    return episode_dir / names[0] / "state.parquet"


def _episode_streams(dataset_dir: Path, episode_id: str, cameras: list[str]):
    episode_dir = dataset_dir / "episodes" / episode_id
    if not episode_dir.is_dir():
        raise SystemExit(f"missing episode directory {episode_dir}")
    cmd_times, cmd_values = _vectors(_stream_file(episode_dir, ("cmd", "action")))
    follower_times, follower_values = _vectors(_stream_file(episode_dir, ("follower", "obs")))
    images = {name: _camera_paths(episode_dir / "cameras" / name) for name in cameras}
    return cmd_times, cmd_values, follower_times, follower_values, images


def _missing(cameras: list[str], cmd_times: list[int], follower_times: list[int], images: dict[str, tuple[list[int], list[Path]]]) -> list[str]:
    missing = []
    if not cmd_times:
        missing.append("cmd")
    if not follower_times:
        missing.append("follower")
    missing.extend(name for name in cameras if not images[name][0])
    return missing


def _convert(dataset: LeRobotDataset | None, dataset_dir: Path, episode: dict, fps: int, joints: list[str], cameras: list[str], fallback_task: str | None, root: Path, repo_id: str, robot_type: str, skip_incomplete: bool) -> LeRobotDataset | None:
    episode_id = str(episode.get("id"))
    cmd_times, cmd_values, follower_times, follower_values, images = _episode_streams(dataset_dir, episode_id, cameras)
    missing = _missing(cameras, cmd_times, follower_times, images)
    if missing:
        message = f"episode {episode_id} is missing {', '.join(missing)}"
        if skip_incomplete:
            print(f"{message}; skipping")
            return dataset
        raise SystemExit(message)
    for name, values in (("cmd", cmd_values), ("follower", follower_values)):
        width = values[0].size
        if width != len(joints):
            raise SystemExit(f"episode {episode_id} {name} has {width} values; expected {len(joints)} joints")
    shapes = {name: _decode(paths[0]).shape[:2] for name, (_times, paths) in images.items()}
    task = episode.get("task") or fallback_task
    if not task:
        raise SystemExit(f"episode {episode_id} has no task; pass --task")
    streams = {"cmd": cmd_times, "follower": follower_times, **{name: times for name, (times, _paths) in images.items()}}
    moments = _moments(streams, fps)
    if dataset is None:
        dataset = _open_dataset(root, repo_id, fps, robot_type, joints, shapes)
    for moment_ns in moments:
        frame: dict = {
            "task": str(task),
            "action": cmd_values[_at(cmd_times, moment_ns)],
            "observation.state": follower_values[_at(follower_times, moment_ns)],
        }
        for name, (times, paths) in images.items():
            image = _decode(paths[_at(times, moment_ns)])
            height, width = shapes[name]
            if image.shape[:2] != (height, width):
                raise SystemExit(f"episode {episode_id} {name} is {image.shape[1]}x{image.shape[0]}; expected {width}x{height}")
            frame[f"observation.images.{name}"] = image
        dataset.add_frame(frame)
    dataset.save_episode()
    print(f"episode {episode_id}: {len(moments)} frames ({len(moments) / fps:.3f} s) task={task}")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a record dataset into a LeRobot dataset")
    parser.add_argument("dataset", type=Path, help="record dataset directory, e.g. record/so101")
    parser.add_argument("--root", type=Path, default=Path("datasets/so101"), help="LeRobot dataset directory to create or append to")
    parser.add_argument("--repo-id", default="local/so101")
    parser.add_argument("--task", default=None, help="task text used when an episode does not have one")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default=None)
    parser.add_argument("--episodes", nargs="*", help="episode ids to convert; default is every finished episode")
    parser.add_argument("--success-only", action="store_true", help="convert only episodes marked success")
    parser.add_argument("--skip-incomplete", action="store_true", help="skip episodes missing a camera, cmd, or follower stream")
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    metadata = _metadata(args.dataset)
    joints, cameras, robot_type = _names(metadata)
    if args.robot_type:
        robot_type = args.robot_type
    if not cameras:
        first = args.dataset / "episodes" / str(_selected(metadata, args.episodes, args.success_only)[0].get("id")) / "cameras"
        cameras = sorted(path.name for path in first.iterdir() if path.is_dir()) if first.is_dir() else []
    if not cameras:
        raise SystemExit("no cameras found in metadata.yaml or the episode directories")

    dataset: LeRobotDataset | None = None
    try:
        for episode in _selected(metadata, args.episodes, args.success_only):
            dataset = _convert(
                dataset,
                args.dataset,
                episode,
                args.fps,
                joints,
                cameras,
                args.task,
                args.root,
                args.repo_id,
                robot_type,
                args.skip_incomplete,
            )
    finally:
        if dataset is not None:
            dataset.finalize()
    if dataset is None:
        raise SystemExit("no episodes were written")
    print(f"wrote {args.root}")


if __name__ == "__main__":
    main()
