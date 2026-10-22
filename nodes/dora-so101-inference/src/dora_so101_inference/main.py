import os
import queue
import threading

import numpy as np
import pyarrow as pa
import torch
from dora import Node
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE


def _feature_key(input_id: str, keys: list[str]) -> str | None:
    for key in keys:
        if key == input_id or key.endswith(f".{input_id}"):
            return key
    return None


def _steps(config) -> int | None:
    raw = os.environ.get("ACTIONS_PER_CHUNK")
    if raw:
        return int(raw)
    for name in ("n_action_steps", "chunk_size"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _load():
    path = os.environ.get("POLICY")
    if not path:
        raise RuntimeError("POLICY is required")
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    config = PreTrainedConfig.from_pretrained(path)
    policy = get_policy_class(config.type).from_pretrained(path)
    policy.to(device)
    policy.eval()
    device_override = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=path,
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
    )
    print(f"policy {config.type} loaded on {device}", flush=True)
    return policy, preprocessor, postprocessor, device


def _vector(action) -> list[float]:
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action)
    return action.detach().float().cpu().reshape(-1).tolist()


def _store(event: dict, latest: dict, image_keys: list[str]) -> None:
    row = event["value"][0] if isinstance(event["value"], pa.StructArray) else event["value"]
    metadata = event["metadata"]
    latest[OBS_STATE] = np.asarray(row["state"].values, dtype=np.float32).reshape(-1).copy()
    for name in row.type.names:
        key = _feature_key(name, image_keys)
        if key is None:
            continue
        height = int(metadata[f"{name}.height"])
        width = int(metadata[f"{name}.width"])
        flat = np.asarray(row[name].values, dtype=np.uint8)
        latest[key] = flat.reshape(height, width, 3).copy()
    task = row["task"].as_py()
    if task:
        latest["task"] = task


def _ready(latest: dict, image_keys: list[str], needs_state: bool) -> bool:
    if image_keys and not any(key in latest for key in image_keys):
        return False
    if needs_state and OBS_STATE not in latest:
        return False
    return True


def _predict(policy, preprocessor, postprocessor, latest: dict, image_keys: list[str], needs_state: bool):
    present = [key for key in image_keys if key in latest]
    if not _ready(latest, image_keys, needs_state):
        return []
    observation = {}
    for key in present:
        image = torch.from_numpy(np.ascontiguousarray(latest[key]))
        image = image.permute(2, 0, 1).to(dtype=torch.float32).div_(255).unsqueeze(0)
        observation[key] = image
    if needs_state:
        observation[OBS_STATE] = torch.tensor(latest[OBS_STATE], dtype=torch.float32).unsqueeze(0)
    task = latest.get("task") or os.environ.get("TASK")
    if task:
        observation["task"] = task
    observation = preprocessor(observation)
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(observation)
    if chunk.ndim == 2:
        chunk = chunk.unsqueeze(0)
    steps = _steps(policy.config)
    count = chunk.shape[1] if steps is None else min(steps, chunk.shape[1])
    return [_vector(postprocessor(chunk[:, index, :])) for index in range(count)]


def main() -> None:
    node = Node()
    policy, preprocessor, postprocessor, _device = _load()
    image_keys = list(policy.config.image_features)
    needs_state = OBS_STATE in policy.config.input_features
    latest: dict = {}
    pending: list[list[float]] = []
    lock = threading.Lock()
    running = False
    jobs: queue.Queue = queue.Queue()

    def _worker() -> None:
        nonlocal running
        while True:
            job = jobs.get()
            if job is None:
                return
            snapshot, drop = job
            try:
                actions = _predict(
                    policy, preprocessor, postprocessor, snapshot, image_keys, needs_state
                )
            except Exception as exc:
                print(f"inference failed ({exc}); skipping", flush=True)
                actions = []
            actions = actions[drop:]
            with lock:
                pending.extend(actions)
                running = False
            if actions:
                print(f"chunk {len(actions)} (skipped {drop})", flush=True)

    threading.Thread(target=_worker, daemon=True).start()
    for event in node:
        if event["type"] == "STOP":
            jobs.put(None)
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "observation":
            try:
                with lock:
                    _store(event, latest, image_keys)
            except Exception as exc:
                print(f"observation rejected ({exc}); skipping", flush=True)
            continue
        if event["id"] == "tick":
            action = None
            job = None
            with lock:
                if pending:
                    action = pending.pop(0)
                if not running and len(pending) <= 35 and _ready(latest, image_keys, needs_state):
                    running = True
                    drop = len(pending) + (1 if action is not None else 0)
                    job = (dict(latest), drop)
            if job is not None:
                jobs.put(job)
            if action is not None:
                node.send_output("action", pa.array(action, type=pa.float32()))


if __name__ == "__main__":
    main()
