"""Turn `.drec` recordings into one LeRobot dataset.

Each file becomes one episode. Frames are sampled on a fixed clock. At each
tick the frame keeps the latest leader command, follower state, and camera
image that had already arrived. Cameras are encoded straight to MP4.

    uv run tools/drec2lerobot.py record/test1 --root datasets/so101 --task "Pick up the cube"
"""

from __future__ import annotations

import argparse
import bisect
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MAGIC = b"DORAREC\x00"
FOOTER_MAGIC = b"DORAEND\x00"
FORMAT_VERSION = 2
ARROW_IPC = b"\xff\xff\xff\xff"

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
LEADER = "leader/leader_cmd"
FOLLOWER = "follower/follower_state"
CAMERAS = {
    "wrist": "wrist-camera/image",
    "overhead": "overhead-camera/image",
}


@dataclass(frozen=True)
class Entry:
    topic: str
    offset_ns: int
    values: np.ndarray


def _read_entries(path: Path) -> list[Entry]:
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise SystemExit(f"{path} is not a .drec file")
    version = struct.unpack_from("<H", data, 8)[0]
    if version != FORMAT_VERSION:
        raise SystemExit(f"{path} uses format version {version}; this tool reads version {FORMAT_VERSION}")
    yaml_len = struct.unpack_from("<I", data, 34)[0]
    pos = 38 + yaml_len
    entries: list[Entry] = []
    while pos + 8 <= len(data):
        if data[pos : pos + 8] == FOOTER_MAGIC:
            break
        record_len = struct.unpack_from("<I", data, pos)[0]
        body = data[pos + 4 : pos + 4 + record_len]
        if record_len < 12 or pos + 4 + record_len > len(data):
            raise SystemExit(f"{path} is truncated at byte {pos}")
        node_len = struct.unpack_from("<H", body, 0)[0]
        node = body[2 : 2 + node_len].decode()
        output_at = 2 + node_len
        output_len = struct.unpack_from("<H", body, output_at)[0]
        output = body[output_at + 2 : output_at + 2 + output_len].decode()
        time_at = output_at + 2 + output_len
        offset_ns = struct.unpack_from("<Q", body, time_at)[0]
        payload_len = struct.unpack_from("<I", body, time_at + 8)[0]
        payload = body[time_at + 12 : time_at + 12 + payload_len]
        entries.append(Entry(f"{node}/{output}", offset_ns, _values(payload)))
        pos += 4 + record_len
    return entries


def _values(payload: bytes) -> np.ndarray:
    ipc_at = payload.find(ARROW_IPC)
    if ipc_at < 0:
        raise SystemExit("Arrow payload not found")
    batch = pa.ipc.open_stream(payload[ipc_at:]).read_next_batch()
    return batch.column(0).to_numpy(zero_copy_only=False)


def _by_topic(entries: list[Entry]) -> dict[str, list[Entry]]:
    grouped: dict[str, list[Entry]] = {}
    for entry in entries:
        grouped.setdefault(entry.topic, []).append(entry)
    for topic in grouped:
        grouped[topic].sort(key=lambda entry: entry.offset_ns)
    return grouped


def _latest(entries: list[Entry], moment_ns: int) -> Entry | None:
    times = [entry.offset_ns for entry in entries]
    index = bisect.bisect_right(times, moment_ns) - 1
    if index < 0:
        return None
    return entries[index]


def _image(values: np.ndarray, height: int, width: int) -> np.ndarray:
    header = values[:8].tobytes() if values.size >= 8 else b""
    if header.startswith(b"\xff\xd8") or header.startswith(b"\x89PNG"):
        frame = cv2.imdecode(values, cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit("camera payload could not be decoded")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if values.size == height * width * 3:
        return values.reshape(height, width, 3)
    raise SystemExit(f"camera payload has {values.size} bytes; expected a JPEG or {height}x{width} RGB")


def _joints(entry: Entry) -> np.ndarray:
    values = np.asarray(entry.values, dtype=np.float32).reshape(-1)
    if values.size != len(JOINTS):
        raise SystemExit(f"{entry.topic} has {values.size} values; expected {len(JOINTS)} joints")
    return values


def _moments(grouped: dict[str, list[Entry]], fps: int) -> list[int]:
    topics = [LEADER, FOLLOWER, *CAMERAS.values()]
    missing = [topic for topic in topics if not grouped.get(topic)]
    if missing:
        raise SystemExit(f"recording is missing {', '.join(missing)}")
    start = max(grouped[topic][0].offset_ns for topic in topics)
    end = min(grouped[topic][-1].offset_ns for topic in topics)
    if end < start:
        raise SystemExit("camera and joint streams do not overlap")
    step = round(1_000_000_000 / fps)
    return list(range(start, end + 1, step))


def _features(height: int, width: int) -> dict:
    names = list(JOINTS)
    vector = {"dtype": "float32", "shape": (len(JOINTS),), "names": names}
    features: dict = {"action": vector, "observation.state": vector}
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _recordings(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.drec")))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"{path} does not exist")
    if not files:
        raise SystemExit("no .drec files found")
    return files


def _open_dataset(root: Path, repo_id: str, fps: int, robot_type: str, height: int, width: int) -> LeRobotDataset:
    info = root / "meta" / "info.json"
    if info.is_file():
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=root, streaming_encoding=True)
        if dataset.meta.fps != fps:
            raise SystemExit(f"{root} was recorded at {dataset.meta.fps} fps; this conversion uses {fps}")
        image_keys = [key for key, feature in dataset.meta.features.items() if feature["dtype"] == "image"]
        if image_keys:
            raise SystemExit(f"{root} stores {', '.join(image_keys)} as images; this conversion writes MP4 only")
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
        features=_features(height, width),
    )


def _convert(dataset: LeRobotDataset | None, path: Path, fps: int, task: str, height: int, width: int, root: Path, repo_id: str, robot_type: str) -> LeRobotDataset:
    grouped = _by_topic(_read_entries(path))
    moments = _moments(grouped, fps)
    for index, moment_ns in enumerate(moments):
        images = {
            name: _image(_latest(grouped[topic], moment_ns).values, height, width) for name, topic in CAMERAS.items()
        }
        frame_height, frame_width = images["wrist"].shape[:2]
        if images["overhead"].shape[:2] != (frame_height, frame_width):
            raise SystemExit(f"{path} wrist and overhead resolutions differ")
        if dataset is None:
            dataset = _open_dataset(root, repo_id, fps, robot_type, frame_height, frame_width)
        frame = {
            "task": task,
            "action": _joints(_latest(grouped[LEADER], moment_ns)),
            "observation.state": _joints(_latest(grouped[FOLLOWER], moment_ns)),
        }
        for name, image in images.items():
            if image.shape[:2] != (frame_height, frame_width):
                raise SystemExit(f"{path} {name} is {image.shape[1]}x{image.shape[0]}; expected {frame_width}x{frame_height}")
            frame[f"observation.images.{name}"] = image
        dataset.add_frame(frame)
        if index == 0 and (frame_height, frame_width) != (height, width):
            print(f"using image size {frame_width}x{frame_height} from {path.name}")
    dataset.save_episode()
    print(f"{path}: {len(moments)} frames ({len(moments) / fps:.3f} s)")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .drec recordings into a LeRobot dataset")
    parser.add_argument("recordings", type=Path, nargs="+", help="episode files or a directory of .drec files")
    parser.add_argument("--root", type=Path, default=Path("datasets/so101"), help="dataset directory to create or append to")
    parser.add_argument("--repo-id", default="local/so101")
    parser.add_argument("--task", default="Pick up the cube and place it in the box")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="so101")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    dataset: LeRobotDataset | None = None
    try:
        for path in _recordings(args.recordings):
            dataset = _convert(
                dataset,
                path,
                args.fps,
                args.task,
                args.height,
                args.width,
                args.root,
                args.repo_id,
                args.robot_type,
            )
    finally:
        if dataset is not None:
            dataset.finalize()
    print(f"wrote {args.root}")


if __name__ == "__main__":
    main()
