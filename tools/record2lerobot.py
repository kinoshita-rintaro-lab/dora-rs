"""Turn a dora-so101-record dataset into one LeRobot dataset.

Frames come from tools/curation.py. The reference camera is the clock. Other
cameras and the leader command are the nearest sample within --slop. Follower
joints are interpolated at that time. Reference frames that fall inside
another camera's gap are left out. The recording stays one episode. The
command becomes the LeRobot action and the follower measurement becomes
observation.state. Both are joint angles.
Cameras are encoded to MP4. meta/conversion.json records that the frames
were curated, plus the fps, the slop, and the record episode ids already
written. A later run skips those ids and appends only new episodes.

An episode that is missing a camera from the recording setup is an error.
Pass --skip-incomplete to leave those episodes out. Episodes whose success
flag is still unset are skipped.

    uv run tools/record2lerobot.py record/pick-and-place --root datasets/so101 --reference overhead
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curation  # noqa: E402

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


def _decode(path: Path) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"could not decode {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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


def _conversion(fps: int, slop: float) -> dict:
    return {
        "curated": True,
        "fps": fps,
        "slop_s": slop,
        "action": "joint_angles",
        "state": "joint_angles",
    }


def _total_episodes(root: Path) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 0
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info.get("total_episodes") or 0)


def _source_episodes(root: Path, fps: int, slop: float) -> list[str]:
    path = root / "meta" / "conversion.json"
    total = _total_episodes(root)
    if not path.is_file():
        if total:
            raise SystemExit(f"{root} has {total} episodes but no {path}, so already-converted episodes cannot be skipped")
        return []
    stored = json.loads(path.read_text(encoding="utf-8"))
    expected = _conversion(fps, slop)
    recorded = {key: stored.get(key) for key in expected}
    if recorded != expected:
        raise SystemExit(f"{path} records {recorded}; this conversion is {expected}")
    if "source_episodes" not in stored:
        if total:
            raise SystemExit(f"{path} has no source_episodes, so already-converted episodes cannot be skipped")
        return []
    source = [str(episode_id) for episode_id in stored["source_episodes"]]
    if len(source) != total:
        raise SystemExit(f"{path} lists {len(source)} source episodes; {root} has {total}")
    return source


def _write_conversion(root: Path, fps: int, slop: float, source_episodes: list[str]) -> None:
    path = root / "meta" / "conversion.json"
    payload = _conversion(fps, slop)
    payload["source_episodes"] = source_episodes
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def _problem(message: str, skip_incomplete: bool) -> None:
    if skip_incomplete:
        print(f"{message}; skipping")
        return
    raise SystemExit(message)


def _convert(dataset: LeRobotDataset | None, dataset_dir: Path, episode: dict, fps: int, joints: list[str], cameras: list[str], fallback_task: str | None, root: Path, repo_id: str, robot_type: str, skip_incomplete: bool, reference: str, slop: float, max_gap: float) -> tuple[LeRobotDataset | None, bool]:
    episode_id = str(episode.get("id"))
    episode_dir = dataset_dir / "episodes" / episode_id
    if not episode_dir.is_dir():
        raise SystemExit(f"missing episode directory {episode_dir}")
    if reference not in cameras:
        raise SystemExit(f"reference camera {reference!r} is not in {', '.join(cameras)}")
    try:
        synced = curation.sync_episode(episode_dir, reference, slop)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    missing = [name for name in cameras if len(synced["camera_times"].get(name, ())) == 0]
    if synced["action"].shape[-1] != len(joints):
        missing.append("cmd")
    if synced["state"].shape[-1] != len(joints):
        missing.append("follower")
    if missing:
        _problem(f"episode {episode_id} is missing {', '.join(missing)}", skip_incomplete)
        return dataset, False
    task = episode.get("task") or fallback_task
    if not task:
        raise SystemExit(f"episode {episode_id} has no task; pass --task")
    synced = curation.drop_gap_frames(synced, max_gap)
    if len(synced["t"]) == 0:
        _problem(f"episode {episode_id} kept no frames", skip_incomplete)
        return dataset, False
    paths = synced["camera_paths"]
    if dataset is None:
        shapes = {name: _decode(paths[name][int(synced["camera_index"][name][0])]).shape[:2] for name in cameras}
        dataset = _open_dataset(root, repo_id, fps, robot_type, joints, shapes)
    else:
        shapes = {name: _shape(dataset.meta.features[f"observation.images.{name}"])[:2] for name in cameras}
    for row in range(len(synced["t"])):
        frame: dict = {
            "task": str(task),
            "action": synced["action"][row],
            "observation.state": synced["state"][row],
        }
        for name in cameras:
            image = _decode(paths[name][int(synced["camera_index"][name][row])])
            height, width = shapes[name]
            if image.shape[:2] != (height, width):
                raise SystemExit(f"episode {episode_id} {name} is {image.shape[1]}x{image.shape[0]}; expected {width}x{height}")
            frame[f"observation.images.{name}"] = image
        dataset.add_frame(frame)
    dataset.save_episode()
    print(f"episode {episode_id}: {len(synced['t'])} frames ({len(synced['t']) / fps:.3f} s) task={task}")
    return dataset, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a record dataset into a LeRobot dataset")
    parser.add_argument("dataset", type=Path, help="record dataset directory, e.g. record/so101")
    parser.add_argument("--root", type=Path, default=Path("datasets/so101"), help="LeRobot dataset directory to create or append to")
    parser.add_argument("--repo-id", default="local/so101")
    parser.add_argument("--task", default=None, help="task text used when an episode does not have one")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--reference", required=True, help="camera whose timestamps are the master clock, e.g. overhead")
    parser.add_argument("--slop", type=float, default=0.020, help="nearest-match window in seconds (default: 0.020)")
    parser.add_argument("--max-gap", type=float, default=0.050, help="drop reference frames inside another camera's gap longer than this (default: 0.050)")
    parser.add_argument("--robot-type", default=None)
    parser.add_argument("--episodes", nargs="*", help="episode ids to convert; default is every finished episode")
    parser.add_argument("--success-only", action="store_true", help="convert only episodes marked success")
    parser.add_argument("--skip-incomplete", action="store_true", help="skip episodes missing a camera, cmd, or follower stream")
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.slop < 0:
        raise SystemExit("--slop must be non-negative")
    if args.max_gap <= 0:
        raise SystemExit("--max-gap must be positive")

    metadata = _metadata(args.dataset)
    joints, cameras, robot_type = _names(metadata)
    if args.robot_type:
        robot_type = args.robot_type
    if not cameras:
        first = args.dataset / "episodes" / str(_selected(metadata, args.episodes, args.success_only)[0].get("id")) / "cameras"
        cameras = sorted(path.name for path in first.iterdir() if path.is_dir()) if first.is_dir() else []
    if not cameras:
        raise SystemExit("no cameras found in metadata.yaml or the episode directories")

    converted = _source_episodes(args.root, args.fps, args.slop)
    converted_ids = set(converted)
    pending = []
    for episode in _selected(metadata, args.episodes, args.success_only):
        episode_id = str(episode.get("id"))
        if episode_id in converted_ids:
            print(f"episode {episode_id}: already converted; skipping")
            continue
        pending.append(episode)
    if not pending:
        print(f"no new episodes; {args.root} already has {len(converted)}")
        return

    dataset: LeRobotDataset | None = None
    try:
        for episode in pending:
            dataset, saved = _convert(
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
                args.reference,
                args.slop,
                args.max_gap,
            )
            if saved:
                episode_id = str(episode.get("id"))
                converted.append(episode_id)
                converted_ids.add(episode_id)
                _write_conversion(args.root, args.fps, args.slop, converted)
    finally:
        if dataset is not None:
            dataset.finalize()
    if dataset is None:
        raise SystemExit("no episodes were written")
    print(f"wrote {args.root}")


if __name__ == "__main__":
    main()
