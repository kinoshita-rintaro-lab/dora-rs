import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyarrow as pa
from dora import Node


def _names(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def _decode_jpeg(encoded: np.ndarray) -> np.ndarray:
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _build(observation: dict, cameras: list[str], task: str | None, metadata: dict, pool: ThreadPoolExecutor):
    arrays = []
    names = []
    state = observation["state"]["value"].to_numpy().reshape(-1).astype(np.float32, copy=False)
    arrays.append(pa.array([state], type=pa.list_(pa.float32())))
    names.append("state")

    decoded = {
        name: pool.submit(_decode_jpeg, observation[name]["value"].to_numpy())
        for name in cameras
    }
    for name in cameras:
        image = decoded[name].result()
        metadata[f"{name}.encoding"] = "rgb8"
        metadata[f"{name}.height"] = image.shape[0]
        metadata[f"{name}.width"] = image.shape[1]
        arrays.append(pa.array([image.ravel()], type=pa.list_(pa.uint8())))
        names.append(name)
    arrays.append(pa.array([task], type=pa.string()))
    names.append("task")
    arrays.append(pa.array([observation["id"]], type=pa.int64()))
    names.append("id")
    return pa.StructArray.from_arrays(arrays, names)


def main() -> None:
    cameras = _names("CAMERAS", "wrist,overhead")
    workers = int(os.environ.get("OBSERVER_DECODE_WORKERS", "2"))
    if workers <= 0:
        raise RuntimeError("OBSERVER_DECODE_WORKERS must be positive")
    node = Node()
    observation = {name: None for name in (*cameras, "state")}
    observation["id"] = 0
    task = None
    cv2.setNumThreads(1)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="observer-jpeg")
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            event_id = event["id"]
            if event_id == "tick":
                if any(observation[name] is None for name in (*cameras, "state")):
                    continue
                metadata = {"timestamp": time.time_ns()}
                try:
                    output = _build(observation, cameras, task, metadata, pool)
                except Exception as exc:
                    print(f"observation build failed ({exc}); skipping", flush=True)
                    continue
                node.send_output("observation", output, metadata)
                observation["id"] += 1
                continue
            if event_id == "task":
                task = event["value"][0].as_py()
                continue
            if event_id in observation:
                observation[event_id] = event
    finally:
        pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
