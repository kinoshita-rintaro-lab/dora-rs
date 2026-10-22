# Copyright 2026 Enactic, Inc.
# Copyright 2026 madamadamadao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to record SO-101 data as timestamped per-stream episode files.

Adapted for SO-101 from Enactic's OpenArm dataset recorder (Apache-2.0):
each input stream is collected independently with its own timestamps,
cameras are written as image files during the episode, and joint streams
are flushed to Parquet when the episode finishes.
"""

import copy
import datetime
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from dora import Node

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _names(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def _timestamp_ns(event: dict, *, state: bool = False) -> int:
    metadata = event.get("metadata") or {}
    value = None
    if state and "observation_timestamp" in metadata:
        value = metadata["observation_timestamp"]
    elif "timestamp" in metadata:
        value = metadata["timestamp"]
    if value is None:
        return time.time_ns()
    if isinstance(value, datetime.datetime):
        return math.ceil(value.timestamp() * 1_000_000_000)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return time.time_ns()
    return int(value)


@dataclass
class Episode:
    """Episode related data, one independent list per stream."""

    number: int = 0
    success: bool = False
    task_index: int = 0
    task: str | None = None
    action_timestamps: list[int] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    state_timestamps: list[int] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    camera_frames: dict[str, int] = field(default_factory=dict)


class EpisodeWriter:
    """Writer for one episode."""

    def __init__(self, directory: Path, episode: Episode):
        self._episode = episode
        self._base_directory = directory / "episodes" / str(episode.number)

    def write_camera_image(self, name: str, image: pa.Array, timestamp: int, encoding: str) -> None:
        extension = str(encoding).lower().lstrip(".") or "bin"
        output_path = self._base_directory / "cameras" / name / f"{timestamp}.{extension}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(np.asarray(image.to_numpy()).tobytes())
        self._episode.camera_frames[name] = self._episode.camera_frames.get(name, 0) + 1

    def finish(self) -> None:
        if self._episode.actions:
            self._write_values(
                self._base_directory / "action" / "state.parquet",
                self._episode.action_timestamps,
                self._episode.actions,
            )
        if self._episode.states:
            self._write_values(
                self._base_directory / "obs" / "state.parquet",
                self._episode.state_timestamps,
                self._episode.states,
            )

    def cancel(self) -> None:
        shutil.rmtree(self._base_directory, ignore_errors=True)

    def _write_values(self, output_path: Path, timestamps: list[int], values: list[np.ndarray]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
                "value": pa.array(values, type=pa.list_(pa.float32())),
            }
        )
        pq.write_table(table, output_path)


class DatasetWriter:
    """Write a dataset directory and its metadata.yaml."""

    _VERSION = "0.1.0"

    def __init__(self, directory: Path, name: str, metadata: dict):
        self._base_directory = directory / name
        self._metadata = metadata
        self._episode_results: list[dict] = []
        if self._base_directory.exists():
            existing = self._read_existing_metadata()
            if existing is not None:
                mismatched = {
                    key: value
                    for key, value in self._metadata.items()
                    if key in existing and existing[key] != value
                }
                if mismatched:
                    raise ValueError(
                        f"Existing dataset metadata does not match: {self._base_directory / 'metadata.yaml'}\n"
                        + "\n".join(
                            f"  {key}: existing={existing.get(key)!r}, current={value!r}"
                            for key, value in mismatched.items()
                        )
                    )
        else:
            self._base_directory.mkdir(parents=True)

    def next_episode_number(self) -> int:
        numbers = []
        for result in self._episode_results:
            try:
                numbers.append(int(result["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        episodes_directory = self._base_directory / "episodes"
        if episodes_directory.is_dir():
            for child in episodes_directory.iterdir():
                if child.is_dir() and child.name.isdigit():
                    numbers.append(int(child.name))
        return max(numbers, default=-1) + 1

    def create_episode_writer(self, episode: Episode) -> EpisodeWriter:
        episode_id = str(episode.number)
        if episode_id in {result["id"] for result in self._episode_results}:
            raise ValueError(f"Episode {episode_id} already exists in dataset")
        episode_directory = self._base_directory / "episodes" / episode_id
        if episode_directory.exists():
            raise ValueError(f"Episode directory already exists: {episode_directory}")
        return EpisodeWriter(self._base_directory, episode)

    def finish_episode(self, episode: Episode) -> None:
        result = {
            "id": str(episode.number),
            "success": episode.success,
            "task_index": episode.task_index,
            "frames": {
                "action": len(episode.actions),
                "state": len(episode.states),
                "cameras": dict(episode.camera_frames),
            },
        }
        if episode.task is not None:
            result["task"] = episode.task
        self._episode_results.append(result)
        self._write_metadata_file()

    def _write_metadata_file(self) -> None:
        metadata = copy.deepcopy(self._metadata)
        metadata["version"] = self._VERSION
        metadata["episodes"] = self._episode_results
        output_path = self._base_directory / "metadata.yaml"
        with output_path.open("w", encoding="utf-8") as output:
            yaml.dump(metadata, output, default_flow_style=False, allow_unicode=True)

    def _read_existing_metadata(self) -> dict | None:
        metadata_path = self._base_directory / "metadata.yaml"
        if not metadata_path.exists():
            return None
        with metadata_path.open(encoding="utf-8") as source:
            existing_metadata = yaml.safe_load(source) or {}
        self._episode_results = existing_metadata.get("episodes", [])
        return existing_metadata


def _timer_frequency(value, nodes: dict, seen: frozenset) -> float | None:
    if not isinstance(value, str):
        if isinstance(value, dict):
            value = value.get("source")
        if not isinstance(value, str):
            return None
    if not value.startswith("dora/timer/"):
        node_id = value.split("/", 1)[0]
        if node_id in seen:
            return None
        config = nodes.get(node_id) or {}
        inputs = config.get("inputs") or {}
        next_input = inputs.get("tick") or inputs.get("request_state") or inputs.get("request_position")
        if next_input is None:
            return None
        return _timer_frequency(next_input, nodes, seen | {node_id})
    parts = value.split("/")
    if len(parts) < 4:
        return None
    unit = parts[2]
    try:
        amount = float(parts[3])
    except ValueError:
        return None
    if unit == "secs":
        return 1.0 / amount
    if unit == "millis":
        return 1_000.0 / amount
    if unit == "micros":
        return 1_000_000.0 / amount
    if unit == "nanos":
        return 1_000_000_000.0 / amount
    if unit == "hz":
        return amount
    return None


def _collect_frequencies(node: Node) -> dict:
    """Best-effort input frequencies from the dataflow descriptor."""
    frequencies: dict = {"action": None, "state": None, "cameras": {}}
    try:
        descriptor = node.dataflow_descriptor()
        own = node.node_config()
    except Exception:
        return frequencies
    nodes = {config.get("id"): config for config in descriptor.get("nodes", [])}
    for name, source in (own.get("inputs") or {}).items():
        frequency = _timer_frequency(source, nodes, frozenset())
        if not frequency:
            continue
        if name == "action":
            frequencies["action"] = frequency
        elif name == "state":
            frequencies["state"] = frequency
        elif name.startswith("camera_"):
            frequencies["cameras"][name.removeprefix("camera_")] = frequency
    return frequencies


def _command_value(event: dict) -> str:
    value = event["value"]
    if hasattr(value, "as_py"):
        return str(value[0].as_py())
    return str(value.to_pylist()[0])


def _vector(event: dict, joints: list[str], label: str) -> np.ndarray | None:
    values = np.asarray(event["value"].to_numpy(), dtype=np.float32).reshape(-1)
    if values.size != len(joints):
        print(f"{label} has {values.size} values, expected {len(joints)}; skipping", flush=True)
        return None
    return values.copy()


def main() -> None:
    directory = Path(os.environ.get("DIRECTORY", os.getcwd()))
    name = os.environ.get("NAME", "dataset")
    joints = _names("JOINTS", ",".join(JOINTS))

    node = Node()
    metadata: dict = {}
    metadata_file = os.environ.get("METADATA_FILE")
    if metadata_file:
        with Path(metadata_file).open(encoding="utf-8") as source:
            metadata = yaml.safe_load(source) or {}
    metadata["operation_type"] = os.environ.get("OPERATION_TYPE", "teleop")
    metadata["robot"] = {
        "type": os.environ.get("ROBOT_TYPE", "so101"),
        "joints": joints,
    }
    metadata["frequencies"] = _collect_frequencies(node)
    dataset_writer = DatasetWriter(directory, name, metadata)

    episode: Episode | None = None
    episode_writer: EpisodeWriter | None = None

    if os.environ.get("AUTO_START", "").lower() in ("1", "true", "yes"):
        episode = Episode(
            number=dataset_writer.next_episode_number(),
            task_index=int(os.environ.get("TASK_INDEX", "0")),
            task=os.environ.get("TASK"),
            success=os.environ.get("AUTO_FINISH", "success") == "success",
        )
        episode_writer = dataset_writer.create_episode_writer(episode)
        print(f"episode {episode.number} started (auto)", flush=True)

    def finish_current() -> None:
        nonlocal episode, episode_writer
        if episode is None or episode_writer is None:
            return
        episode_writer.finish()
        dataset_writer.finish_episode(episode)
        print(
            f"episode {episode.number} recorded "
            f"(success={episode.success}, action={len(episode.actions)}, state={len(episode.states)})",
            flush=True,
        )
        episode = None
        episode_writer = None

    try:
        for event in node:
            if event["type"] == "STOP":
                finish_current()
                break
            if event["type"] != "INPUT":
                continue
            event_id = event["id"]

            if event_id == "command":
                command = _command_value(event)
                command_metadata = event.get("metadata") or {}
                if command == "start":
                    if episode is not None:
                        print(f"episode {episode.number} already recording; skipping start", flush=True)
                        continue
                    raw_number = command_metadata.get("episode_number")
                    episode = Episode(
                        number=dataset_writer.next_episode_number() if raw_number is None else int(raw_number),
                        task_index=int(command_metadata.get("task_index", 0)),
                        task=command_metadata.get("task") or os.environ.get("TASK"),
                    )
                    episode_writer = dataset_writer.create_episode_writer(episode)
                    print(f"episode {episode.number} started", flush=True)
                elif command in ("success", "fail"):
                    if episode is None:
                        print(f"{command} received with no episode recording; skipping", flush=True)
                        continue
                    episode.success = command == "success"
                    finish_current()
                elif command == "cancel":
                    if episode is None:
                        print("cancel received with no episode recording; skipping", flush=True)
                        continue
                    episode_writer.cancel()
                    print(f"episode {episode.number} cancelled", flush=True)
                    episode = None
                    episode_writer = None
                elif command == "quit":
                    finish_current()
                    break
                else:
                    print(f"unknown command {command!r}; skipping", flush=True)
                continue

            if episode is None or episode_writer is None:
                continue

            if event_id == "action":
                values = _vector(event, joints, "action")
                if values is not None:
                    episode.actions.append(values)
                    episode.action_timestamps.append(_timestamp_ns(event))
            elif event_id == "state":
                values = _vector(event, joints, "state")
                if values is not None:
                    episode.states.append(values)
                    episode.state_timestamps.append(_timestamp_ns(event, state=True))
            elif event_id.startswith("camera_"):
                camera = event_id.removeprefix("camera_")
                encoding = (event.get("metadata") or {}).get("encoding", "jpeg")
                try:
                    episode_writer.write_camera_image(
                        camera, event["value"], _timestamp_ns(event), str(encoding)
                    )
                except Exception as exc:
                    print(f"camera {camera} write failed ({exc}); skipping", flush=True)
    finally:
        if episode is not None:
            finish_current()


if __name__ == "__main__":
    main()
