"""Align a recorded episode onto one camera's timestamps.

The reference camera is the master clock. Every other camera and the leader
command must have a sample within ``slop`` seconds of that frame; otherwise
the frame is dropped. The follower joint vector is linearly interpolated, and
a frame outside that stream's range is dropped.

    uv run tools/curation.py record/so101 --reference wrist
    uv run tools/curation.py record/so101 --reference overhead --slop 0.020
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def nearest(stamps: np.ndarray, t: int, slop: int) -> int | None:
    """Return the index of the stamp closest to ``t``.

    ``stamps`` is sorted ascending. ``slop`` uses the same unit as ``stamps``.
    The result is None when ``stamps`` is empty or the closest stamp is farther
    than ``slop``.
    """
    i = int(np.searchsorted(stamps, t))
    candidates = [j for j in (i - 1, i) if 0 <= j < len(stamps)]
    if not candidates:
        return None
    best = min(candidates, key=lambda j: abs(int(stamps[j]) - int(t)))
    if abs(int(stamps[best]) - int(t)) > slop:
        return None
    return best


def _metadata(path: Path) -> dict:
    metadata_path = path / "metadata.yaml"
    if not metadata_path.is_file() or not (path / "episodes").is_dir():
        raise SystemExit(f"{path} is not a record dataset (expected metadata.yaml and episodes/)")
    with metadata_path.open(encoding="utf-8") as source:
        metadata = yaml.safe_load(source) or {}
    if not isinstance(metadata, dict) or not isinstance(metadata.get("episodes"), list):
        raise SystemExit(f"{metadata_path} has no episode list")
    return metadata


def _stream_file(episode_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = episode_dir / name / "state.parquet"
        if path.is_file():
            return path
    return episode_dir / names[0] / "state.parquet"


def _vectors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    table = pq.read_table(path)
    times = np.asarray(table.column("timestamp").cast(pa.int64()).to_numpy(), dtype=np.int64)
    columns = [np.asarray(item.as_py(), dtype=np.float32).reshape(-1) for item in table.column("value")]
    values = np.stack(columns) if columns else np.zeros((0, 0), dtype=np.float32)
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    if len(times) > 1:
        keep = np.empty(len(times), dtype=bool)
        keep[-1] = True
        keep[:-1] = times[:-1] != times[1:]
        times, values = times[keep], values[keep]
    return times, values


def _camera_frames(directory: Path) -> tuple[np.ndarray, list[Path]]:
    if not directory.is_dir():
        return np.zeros(0, dtype=np.int64), []
    frames = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            frames.append((int(path.stem), path))
    frames.sort()
    if not frames:
        return np.zeros(0, dtype=np.int64), []
    return np.asarray([stamp for stamp, _ in frames], dtype=np.int64), [path for _, path in frames]


def _episode_cameras(episode_dir: Path) -> dict[str, tuple[np.ndarray, list[Path]]]:
    root = episode_dir / "cameras"
    if not root.is_dir():
        return {}
    return {path.name: _camera_frames(path) for path in sorted(root.iterdir()) if path.is_dir()}


def _interp(times: np.ndarray, values: np.ndarray, t: int) -> np.ndarray:
    origin = int(times[0])
    xp = (times.astype(np.float64) - origin) / 1_000_000_000
    query = (int(t) - origin) / 1_000_000_000
    return np.array([np.interp(query, xp, values[:, joint]) for joint in range(values.shape[1])], dtype=np.float32)


def sync_episode(episode_dir: Path, reference: str, slop: float = 0.020) -> dict:
    """Align one episode onto ``reference`` camera timestamps.

    ``slop`` is the nearest-match window in seconds. Leader command and the
    other cameras use that window. Follower joints are interpolated.
    """
    if slop < 0:
        raise ValueError("slop must be non-negative")
    episode_dir = Path(episode_dir)
    cameras = _episode_cameras(episode_dir)
    if reference not in cameras:
        found = ", ".join(cameras) or "none"
        raise ValueError(f"{episode_dir} has no camera {reference!r}; found {found}")
    slop_ns = round(slop * 1_000_000_000)
    command_times, command_values = _vectors(_stream_file(episode_dir, ("cmd", "action")))
    follower_times, follower_values = _vectors(_stream_file(episode_dir, ("follower", "obs")))
    reference_times, _ = cameras[reference]
    kept_t: list[int] = []
    kept_state: list[np.ndarray] = []
    kept_action: list[np.ndarray] = []
    kept_index: dict[str, list[int]] = {name: [] for name in cameras}
    dropped = {"camera": 0, "action": 0, "state": 0}
    for index, stamp in enumerate(reference_times):
        moment = int(stamp)
        matched = {reference: index}
        missed_camera = False
        for name, (times, _paths) in cameras.items():
            if name == reference:
                continue
            other = nearest(times, moment, slop_ns)
            if other is None:
                missed_camera = True
                break
            matched[name] = other
        if missed_camera:
            dropped["camera"] += 1
            continue
        action_index = nearest(command_times, moment, slop_ns)
        if action_index is None:
            dropped["action"] += 1
            continue
        if len(follower_times) == 0 or not (int(follower_times[0]) <= moment <= int(follower_times[-1])):
            dropped["state"] += 1
            continue
        kept_t.append(moment)
        kept_state.append(_interp(follower_times, follower_values, moment))
        kept_action.append(command_values[action_index])
        for name, frame_index in matched.items():
            kept_index[name].append(frame_index)
    state_width = follower_values.shape[1] if follower_values.size else 0
    action_width = command_values.shape[1] if command_values.size else 0
    return {
        "t": np.asarray(kept_t, dtype=np.int64),
        "state": np.stack(kept_state).astype(np.float32) if kept_state else np.zeros((0, state_width), dtype=np.float32),
        "action": np.stack(kept_action).astype(np.float32) if kept_action else np.zeros((0, action_width), dtype=np.float32),
        "camera_index": {name: np.asarray(indices, dtype=np.int64) for name, indices in kept_index.items()},
        "camera_paths": {name: paths for name, (_times, paths) in cameras.items()},
        "camera_times": {name: times for name, (times, _paths) in cameras.items()},
        "reference": reference,
        "source_frames": int(len(reference_times)),
        "dropped": dropped,
    }


def _slice_synced(synced: dict, indexes: np.ndarray) -> dict:
    return {
        "t": synced["t"][indexes],
        "state": synced["state"][indexes],
        "action": synced["action"][indexes],
        "camera_index": {name: values[indexes] for name, values in synced["camera_index"].items()},
        "camera_paths": synced["camera_paths"],
        "camera_times": synced["camera_times"],
        "reference": synced["reference"],
        "source_frames": synced["source_frames"],
        "dropped": synced["dropped"],
    }


def drop_gap_frames(synced: dict, max_gap: float = 0.050) -> dict:
    """Drop master frames that fall strictly inside another camera's gap.

    The recording stays one episode. No image is created for the missing interval.
    """
    if max_gap <= 0:
        raise ValueError("max_gap must be positive")
    limit = round(max_gap * 1_000_000_000)
    times = synced["t"]
    keep = np.ones(len(times), dtype=bool)
    for name, stamps in synced["camera_times"].items():
        if name == synced["reference"] or len(stamps) < 2:
            continue
        for start, end in zip(stamps[:-1], stamps[1:]):
            if int(end) - int(start) <= limit:
                continue
            keep &= ~((times > int(start)) & (times < int(end)))
    return _slice_synced(synced, np.flatnonzero(keep))


def _selected(metadata: dict, episode_ids: list[str] | None) -> list[dict]:
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
        chosen.append(episode)
    if not chosen:
        raise SystemExit("no episodes to sync")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Align recorded episodes onto one camera clock")
    parser.add_argument("dataset", type=Path, help="record dataset directory, e.g. record/so101")
    parser.add_argument("--reference", required=True, help="camera whose timestamps are the master clock, e.g. wrist")
    parser.add_argument("--slop", type=float, default=0.020, help="nearest-match window in seconds (default: 0.020)")
    parser.add_argument("--max-gap", type=float, default=0.050, help="drop reference frames inside another camera's gap longer than this (default: 0.050)")
    parser.add_argument("--episodes", nargs="*", help="episode ids to sync; default is every episode")
    args = parser.parse_args()
    if args.slop < 0:
        raise SystemExit("--slop must be non-negative")
    if args.max_gap <= 0:
        raise SystemExit("--max-gap must be positive")
    metadata = _metadata(args.dataset)
    for episode in _selected(metadata, args.episodes):
        episode_id = str(episode.get("id"))
        episode_dir = args.dataset / "episodes" / episode_id
        if not episode_dir.is_dir():
            raise SystemExit(f"missing episode directory {episode_dir}")
        try:
            synced = sync_episode(episode_dir, args.reference, args.slop)
            kept = drop_gap_frames(synced, args.max_gap)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        dropped = synced["dropped"]
        print(
            f"episode {episode_id}: kept {len(kept['t'])} / {synced['source_frames']} "
            f"reference={args.reference} slop={args.slop:.3f}s max_gap={args.max_gap:.3f}s "
            f"gap={len(synced['t']) - len(kept['t'])} "
            f"dropped camera={dropped['camera']} action={dropped['action']} state={dropped['state']}"
        )


if __name__ == "__main__":
    main()
