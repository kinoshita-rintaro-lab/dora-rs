"""Open a `.drec` recording in Rerun on the file's shared clock.

Each message stays at its own arrival time. Every joint is its own point
series for leader and for follower, so their timestamps can be compared
one joint at a time.

    uv run tools/view_drec.py record/test1/ep2.drec
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
import rerun as rr
import rerun.blueprint as rrb
from rerun.components import MarkerShape

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
WRIST = "wrist-camera/image"
OVERHEAD = "overhead-camera/image"
# Y value in the timestamp plot. It only keeps the rows from sitting on top of each other.
def _lane(side: str, joint: str | None = None) -> float:
    if joint is None:
        return float(len(JOINTS) * 2 + (0 if side == "wrist" else 1))
    return float(JOINTS.index(joint) + (0 if side == "leader" else len(JOINTS)))


@dataclass(frozen=True)
class Entry:
    topic: str
    offset_ns: int
    values: np.ndarray


def _read_entries(path: Path) -> tuple[int, list[Entry]]:
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise SystemExit(f"{path} is not a .drec file")
    version = struct.unpack_from("<H", data, 8)[0]
    if version != FORMAT_VERSION:
        raise SystemExit(f"{path} uses format version {version}; this tool reads version {FORMAT_VERSION}")
    start_ns = struct.unpack_from("<Q", data, 10)[0]
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
    return start_ns, entries


def _values(payload: bytes) -> np.ndarray:
    ipc_at = payload.find(ARROW_IPC)
    if ipc_at < 0:
        raise SystemExit("Arrow payload not found")
    batch = pa.ipc.open_stream(payload[ipc_at:]).read_next_batch()
    return batch.column(0).to_numpy(zero_copy_only=False)


def _image(values: np.ndarray) -> np.ndarray | None:
    frame = cv2.imdecode(values, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _by_topic(entries: list[Entry]) -> dict[str, list[Entry]]:
    grouped: dict[str, list[Entry]] = {}
    for entry in entries:
        grouped.setdefault(entry.topic, []).append(entry)
    for topic in grouped:
        grouped[topic].sort(key=lambda entry: entry.offset_ns)
    return grouped


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
    return min(candidates, key=lambda time: abs(time - moment_ns))


def _delta_ms(sample_ns: int | None, moment_ns: int) -> float | None:
    if sample_ns is None:
        return None
    return (moment_ns - sample_ns) / 1e6


def _spans(samples: list[tuple[int, bool]]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start: int | None = None
    previous = 0
    for moment_ns, outside in samples:
        if outside and start is None:
            start = moment_ns
        elif not outside and start is not None:
            spans.append((start / 1e9, previous / 1e9))
            start = None
        previous = moment_ns
    if start is not None:
        spans.append((start / 1e9, previous / 1e9))
    return spans


def _report(grouped: dict[str, list[Entry]], threshold_ms: float) -> list[tuple[int, float | None, float | None, float | None]]:
    wrist = grouped.get(WRIST, [])
    leader_times = [entry.offset_ns for entry in grouped.get(LEADER, [])]
    follower_times = [entry.offset_ns for entry in grouped.get(FOLLOWER, [])]
    overhead_times = [entry.offset_ns for entry in grouped.get(OVERHEAD, [])]
    rows: list[tuple[int, float | None, float | None, float | None]] = []
    flags: list[tuple[int, bool]] = []
    for frame in wrist:
        leader_ms = _delta_ms(_previous(leader_times, frame.offset_ns), frame.offset_ns)
        follower_ms = _delta_ms(_previous(follower_times, frame.offset_ns), frame.offset_ns)
        overhead_ms = _delta_ms(_nearest(overhead_times, frame.offset_ns), frame.offset_ns)
        rows.append((frame.offset_ns, leader_ms, follower_ms, overhead_ms))
        outside = any(delta is None or abs(delta) > threshold_ms for delta in (leader_ms, follower_ms, overhead_ms))
        flags.append((frame.offset_ns, outside))

    duration_s = max((entry.offset_ns for topic in grouped.values() for entry in topic), default=0) / 1e9
    print(f"duration {duration_s:.3f} s")
    for topic in (LEADER, FOLLOWER, WRIST, OVERHEAD):
        samples = grouped.get(topic, [])
        if not samples:
            print(f"  {topic}: missing")
            continue
        print(f"  {topic}: {len(samples)} messages, first at {samples[0].offset_ns / 1e9:.3f} s")
    print(f"wrist frames whose delta exceeds {threshold_ms:.0f} ms")
    spans = _spans(flags)
    if not spans:
        print("  none")
    for start_s, end_s in spans:
        print(f"  {start_s:.3f} s to {end_s:.3f} s")
    return rows


def _points(path: str) -> None:
    rr.log(path, rr.SeriesPoints(markers=MarkerShape.Circle, marker_sizes=3), static=True)


def _layout() -> None:
    cameras = rrb.Vertical(
        rrb.Spatial2DView(origin="observation/wrist", name="wrist"),
        rrb.Spatial2DView(origin="observation/overhead", name="overhead"),
    )
    joints = rrb.Vertical(
        *[
            rrb.TimeSeriesView(
                name=joint,
                contents=[f"joints/leader/{joint}", f"joints/follower/{joint}"],
            )
            for joint in JOINTS
        ]
    )
    timestamps = rrb.TimeSeriesView(
        name="timestamps",
        contents=[f"time/leader/{joint}" for joint in JOINTS]
        + [f"time/follower/{joint}" for joint in JOINTS]
        + ["time/wrist", "time/overhead"],
    )
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(cameras, joints, timestamps, column_shares=[2, 3, 3])))


def _log(grouped: dict[str, list[Entry]]) -> None:
    for joint in JOINTS:
        for side in ("leader", "follower"):
            _points(f"joints/{side}/{joint}")
            _points(f"time/{side}/{joint}")
    _points("time/wrist")
    _points("time/overhead")
    for entry in grouped.get(LEADER, []):
        _log_joints("leader", entry)
    for entry in grouped.get(FOLLOWER, []):
        _log_joints("follower", entry)
    for topic, entity, name in (
        (WRIST, "observation/wrist", "wrist"),
        (OVERHEAD, "observation/overhead", "overhead"),
    ):
        for entry in grouped.get(topic, []):
            _log_mark(name, None, entry.offset_ns)
            frame = _image(entry.values)
            if frame is None:
                continue
            rr.set_time("record", duration=entry.offset_ns / 1e9)
            rr.log(entity, rr.Image(frame))


def _log_mark(side: str, joint: str | None, offset_ns: int) -> None:
    path = f"time/{side}" if joint is None else f"time/{side}/{joint}"
    event = f"events/{side}" if joint is None else f"events/{side}/{joint}"
    rr.set_time("record", duration=offset_ns / 1e9)
    rr.log(path, rr.Scalars(_lane(side, joint)))
    # A 2D point is one event. The streams row shows a mark at this timestamp.
    rr.log(event, rr.Points2D([[0.0, _lane(side, joint)]]))


def _log_joints(side: str, entry: Entry) -> None:
    values = entry.values.reshape(-1)
    if values.size != len(JOINTS):
        raise SystemExit(f"{entry.topic} has {values.size} values; expected {len(JOINTS)} joints")
    rr.set_time("record", duration=entry.offset_ns / 1e9)
    for joint, value in zip(JOINTS, values, strict=True):
        rr.log(f"joints/{side}/{joint}", rr.Scalars(float(value)))
        _log_mark(side, joint, entry.offset_ns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a .drec recording in Rerun on the file clock")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--threshold-ms", type=float, default=33.0)
    parser.add_argument("--no-viewer", action="store_true", help="print the out-of-sync spans and exit")
    parser.add_argument("--save", type=Path, help="also write a Rerun .rrd file")
    args = parser.parse_args()
    if not args.recording.is_file():
        raise SystemExit(f"{args.recording} does not exist")

    _start_ns, entries = _read_entries(args.recording)
    grouped = _by_topic(entries)
    _report(grouped, args.threshold_ms)
    if args.no_viewer and args.save is None:
        return

    rr.init(args.recording.stem, spawn=not args.no_viewer)
    _layout()
    _log(grouped)
    if args.save is not None:
        rr.save(args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
