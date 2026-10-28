"""Open one recorded episode in Rerun on the episode clock.

cmd is the leader command and follower is the measured joint state. Each
joint is its own series. Older episodes stored these as action/ and obs/.

    uv run tools/view_record.py record/so101 --episode 0
    uv run tools/view_record.py record/so101/episodes/0
"""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb
import yaml
from rerun.components import MarkerShape

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _metadata(path: Path) -> dict | None:
    metadata_path = path / "metadata.yaml"
    if not metadata_path.is_file():
        return None
    with metadata_path.open(encoding="utf-8") as source:
        metadata = yaml.safe_load(source) or {}
    return metadata if isinstance(metadata, dict) else None


def _episode_list(metadata: dict) -> str:
    lines = []
    for episode in metadata.get("episodes") or []:
        cameras = ", ".join((episode.get("frames") or {}).get("cameras") or []) or "no cameras"
        lines.append(f"  {episode.get('id')}: success={episode.get('success')} {cameras} {episode.get('task') or ''}")
    return "\n".join(lines)


def _resolve(path: Path, episode_id: str | None) -> tuple[Path, list[str], dict | None]:
    metadata = _metadata(path)
    if metadata is not None and (path / "episodes").is_dir():
        episodes = metadata.get("episodes") or []
        if episode_id is None:
            if len(episodes) == 1:
                episode_id = str(episodes[0].get("id"))
            else:
                raise SystemExit(f"pass --episode N\n{_episode_list(metadata)}")
        known = {str(episode.get("id")): episode for episode in episodes}
        if episode_id not in known:
            raise SystemExit(f"unknown episode {episode_id}\n{_episode_list(metadata)}")
        joints = list((metadata.get("robot") or {}).get("joints") or JOINTS)
        return path / "episodes" / episode_id, joints, known[episode_id]
    if any((path / name / "state.parquet").is_file() for name in ("cmd", "follower", "action", "obs")) or (path / "cameras").is_dir():
        return path, list(JOINTS), None
    raise SystemExit(f"{path} is not a record dataset or an episode directory")


def _stream_file(episode_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = episode_dir / name / "state.parquet"
        if path.is_file():
            return path
    return episode_dir / names[0] / "state.parquet"


def _vectors(path: Path) -> tuple[list[int], list[np.ndarray]]:
    if not path.is_file():
        return [], []
    table = pq.read_table(path)
    times = table.column("timestamp").cast(pa.int64()).to_pylist()
    values = [np.asarray(item.as_py(), dtype=np.float32).reshape(-1) for item in table.column("value")]
    order = sorted(range(len(times)), key=times.__getitem__)
    return [times[index] for index in order], [values[index] for index in order]


def _cameras(episode_dir: Path) -> dict[str, tuple[list[int], list[Path]]]:
    root = episode_dir / "cameras"
    cameras: dict[str, tuple[list[int], list[Path]]] = {}
    if not root.is_dir():
        return cameras
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        frames = []
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                frames.append((int(path.stem), path))
        frames.sort()
        cameras[directory.name] = ([stamp for stamp, _ in frames], [path for _, path in frames])
    return cameras


def _decode(path: Path) -> np.ndarray | None:
    frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _previous(times: list[int], moment_ns: int) -> int | None:
    index = bisect.bisect_right(times, moment_ns) - 1
    if index < 0:
        return None
    return times[index]


def _nearest(times: list[int], moment_ns: int) -> int | None:
    if not times:
        return None
    index = bisect.bisect_left(times, moment_ns)
    candidates = []
    if index < len(times):
        candidates.append(times[index])
    if index > 0:
        candidates.append(times[index - 1])
    return min(candidates, key=lambda stamp: abs(stamp - moment_ns))


def _delta_ms(sample_ns: int | None, moment_ns: int) -> float | None:
    if sample_ns is None:
        return None
    return (moment_ns - sample_ns) / 1e6


def _spans(samples: list[tuple[int, bool]]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start: int | None = None
    previous = 0
    origin = samples[0][0] if samples else 0
    for moment_ns, outside in samples:
        if outside and start is None:
            start = moment_ns
        elif not outside and start is not None:
            spans.append(((start - origin) / 1e9, (previous - origin) / 1e9))
            start = None
        previous = moment_ns
    if start is not None and samples:
        spans.append(((start - origin) / 1e9, (previous - origin) / 1e9))
    return spans


def _report(cmd: list[int], follower: list[int], cameras: dict[str, tuple[list[int], list[Path]]], threshold_ms: float, episode: dict | None) -> None:
    if episode is not None:
        print(f"episode {episode.get('id')} success={episode.get('success')} task={episode.get('task') or ''}")
    stamps = [*cmd, *follower, *(stamp for times, _paths in cameras.values() for stamp in times)]
    if not stamps:
        raise SystemExit("episode has no cmd, follower, or camera samples")
    origin = min(stamps)
    print(f"duration {(max(stamps) - origin) / 1e9:.3f} s")
    print(f"  cmd: {len(cmd) or 'missing'}")
    print(f"  follower: {len(follower) or 'missing'}")
    for name, (times, _paths) in cameras.items():
        print(f"  camera {name}: {len(times) or 'missing'}")

    reference = "wrist" if "wrist" in cameras and cameras["wrist"][0] else next((name for name, (times, _) in cameras.items() if times), None)
    if reference is None:
        print("no camera frames to check for sync")
        return
    others = [name for name in cameras if name != reference]
    flags: list[tuple[int, bool]] = []
    for moment_ns in cameras[reference][0]:
        deltas = [
            _delta_ms(_previous(cmd, moment_ns), moment_ns),
            _delta_ms(_previous(follower, moment_ns), moment_ns),
            *[_delta_ms(_nearest(cameras[name][0], moment_ns), moment_ns) for name in others],
        ]
        flags.append((moment_ns, any(delta is None or abs(delta) > threshold_ms for delta in deltas)))
    print(f"{reference} frames whose delta exceeds {threshold_ms:.0f} ms")
    spans = _spans(flags)
    if not spans:
        print("  none")
    for start_s, end_s in spans:
        print(f"  {start_s:.3f} s to {end_s:.3f} s")


def _points(path: str) -> None:
    rr.log(path, rr.SeriesPoints(markers=MarkerShape.Circle, marker_sizes=3), static=True)


def _layout(joints: list[str], cameras: list[str]) -> None:
    views = []
    if cameras:
        views.append(
            rrb.Vertical(*[rrb.Spatial2DView(origin=f"observation/{name}", name=name) for name in cameras])
        )
    if joints:
        views.append(
            rrb.Vertical(
                *[
                    rrb.TimeSeriesView(
                        name=joint,
                        contents=[f"joints/cmd/{joint}", f"joints/follower/{joint}"],
                    )
                    for joint in joints
                ]
            )
        )
    if not views:
        return
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views, column_shares=[2, 3][: len(views)])))


def _log_joints(origin: int, side: str, stamp_ns: int, values: np.ndarray, joints: list[str]) -> None:
    if values.size != len(joints):
        raise SystemExit(f"{side} has {values.size} values; expected {len(joints)} joints")
    rr.set_time("record", duration=(stamp_ns - origin) / 1e9)
    for joint, value in zip(joints, values, strict=True):
        rr.log(f"joints/{side}/{joint}", rr.Scalars(float(value)))


def _log(origin: int, cmd: tuple[list[int], list[np.ndarray]], follower: tuple[list[int], list[np.ndarray]], cameras: dict[str, tuple[list[int], list[Path]]], joints: list[str]) -> None:
    for joint in joints:
        for side in ("cmd", "follower"):
            _points(f"joints/{side}/{joint}")
    for stamp_ns, values in zip(*cmd, strict=True):
        _log_joints(origin, "cmd", stamp_ns, values, joints)
    for stamp_ns, values in zip(*follower, strict=True):
        _log_joints(origin, "follower", stamp_ns, values, joints)
    failed = 0
    for name, (times, paths) in cameras.items():
        for stamp_ns, path in zip(times, paths, strict=True):
            frame = _decode(path)
            if frame is None:
                failed += 1
                continue
            rr.set_time("record", duration=(stamp_ns - origin) / 1e9)
            rr.log(f"observation/{name}", rr.Image(frame))
    if failed:
        print(f"{failed} camera frames could not be decoded")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a recorded episode in Rerun")
    parser.add_argument("path", type=Path, help="record dataset directory or one episode directory")
    parser.add_argument("--episode", help="episode id when path is a dataset directory")
    parser.add_argument("--threshold-ms", type=float, default=33.0)
    parser.add_argument("--no-viewer", action="store_true", help="print the sync report and exit")
    parser.add_argument("--save", type=Path, help="also write a Rerun .rrd file")
    args = parser.parse_args()
    if not args.path.exists():
        raise SystemExit(f"{args.path} does not exist")

    episode_dir, joints, episode = _resolve(args.path, args.episode)
    cmd_times, cmd_values = _vectors(_stream_file(episode_dir, ("cmd", "action")))
    follower_times, follower_values = _vectors(_stream_file(episode_dir, ("follower", "obs")))
    cameras = _cameras(episode_dir)
    if cmd_values and cmd_values[0].size != len(joints):
        joints = [f"joint_{index}" for index in range(cmd_values[0].size)]
    _report(cmd_times, follower_times, cameras, args.threshold_ms, episode)
    if args.no_viewer and args.save is None:
        return

    name = episode_dir.name if episode is None else f"episode-{episode.get('id')}"
    rr.init(name, spawn=not args.no_viewer)
    _layout(joints, list(cameras))
    origin = min([*cmd_times, *follower_times, *(stamp for times, _paths in cameras.values() for stamp in times)])
    _log(origin, (cmd_times, cmd_values), (follower_times, follower_values), cameras, joints)
    if args.save is not None:
        rr.save(args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
