# Copyright 2026 Dmitri Manajev
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

# Modified by Rintaro Kinoshita. Real-time chunking logic added.

from __future__ import annotations

import gc
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import cv2
import numpy as np
import torch
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import (
    FPSTracker, Observation, RemotePolicyConfig, TimedAction, TimedObservation,
    extract_state_from_raw_observation, get_logger, is_image_key,
    make_lerobot_observation)
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STATE

logger = get_logger("inference_engine", log_to_file=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_pretrained_config(path: str) -> dict:
    """Return config.json from a local pretrained folder, or an empty dict."""
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open() as f:
        return json.load(f)


def _decode_jpeg_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to an RGB uint8 numpy array (H, W, 3)."""
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed — invalid JPEG data")
    return np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)  # BGR → RGB


def _decode_compressed_images(raw_obs: dict) -> dict:
    """In-place: replace any ``bytes`` values (JPEG) with decoded numpy arrays.

    Non-bytes values are left unchanged so the function is safe to call
    even when no compressed images are present.
    """
    for key, value in raw_obs.items():
        if isinstance(value, (bytes, bytearray)):
            raw_obs[key] = _decode_jpeg_to_rgb(value)
    return raw_obs


def _raw_observation_to_observation(
    raw_observation: dict,
    lerobot_features: dict[str, dict],
) -> Observation:
    """Convert raw robot observation to policy-ready tensors.

    Unlike the upstream ``raw_observation_to_observation`` this does **not**
    resize images.  VLA policies (SmolVLA, Pi0, …) handle resizing internally
    (e.g. ``resize_with_pad``).  Pre-resizing here would distort the aspect
    ratio and create a train / inference mismatch.
    See: https://github.com/huggingface/lerobot/issues/2475
    """
    lerobot_obs = make_lerobot_observation(raw_observation, lerobot_features)

    image_keys = list(filter(is_image_key, lerobot_obs))
    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}

    # HWC → CHW permute, uint8 → float32 [0,1], add batch dim — no resize
    image_dict = {}
    for key in image_keys:
        img = torch.tensor(lerobot_obs[key]).permute(2, 0, 1)  # (H,W,C) → (C,H,W)
        img = img.to(dtype=torch.float32).div_(255).contiguous()  # [0,255] → [0,1]
        image_dict[key] = img.unsqueeze(0)  # (1,C,H,W)

    if "task" in raw_observation:
        state_dict["task"] = raw_observation["task"]

    return {**state_dict, **image_dict}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig:
    fps: int = 30
    inference_latency: float = 0.033
    obs_queue_timeout: float = 2.0

    @property
    def environment_dt(self) -> float:
        return 1.0 / self.fps


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Transport-agnostic inference pipeline.

    Manages observation queuing, policy loading, action-chunk prediction,
    and FPS tracking for the gRPC server.
    """

    def __init__(self, config: InferenceEngineConfig | None = None) -> None:
        self.config = config or InferenceEngineConfig()
        self.shutdown_event = threading.Event()

        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        self.observation_queue: Queue[TimedObservation] = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps: set[int] = set()
        self.last_processed_obs: TimedObservation | None = None

        # Policy state — set by load_policy()
        self.device: str | None = None
        self.policy_type: str | None = None
        self.lerobot_features: dict[str, Any] | None = None
        self.actions_per_chunk: int | None = None
        self.policy: Any = None
        self.preprocessor: PolicyProcessorPipeline | None = None
        self.postprocessor: PolicyProcessorPipeline | None = None
        self._loaded_name_or_path: str | None = None
        self._loaded_rename_map: dict[str, str] | None = None
        # Set by load_pinned_policy(). Client model paths are ignored while this is set.
        self._pinned_name_or_path: str | None = None
        # Previous chunk in normalized space, before postprocess. Used only when the
        # client sends rtc_inference_delay.
        self._rtc_prev: torch.Tensor | None = None
        self._rtc_prev_start: int | None = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()
    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def unload_policy(self) -> None:
        """Release the loaded policy and related processor state."""
        had_policy = any(
            component is not None
            for component in (self.policy, self.preprocessor, self.postprocessor)
        )

        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self._loaded_name_or_path = None
        self._loaded_rename_map = None
        self._rtc_prev = None
        self._rtc_prev_start = None

        if had_policy:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Policy unloaded and caches released")

    def clear_session(self) -> None:
        """Flush the in-flight observation and action state. The loaded weights stay resident."""
        self.observation_queue = Queue(maxsize=1)
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
        self.last_processed_obs = None
        self._rtc_prev = None
        self._rtc_prev_start = None
        self.fps_tracker.reset()
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def reset(self) -> None:
        """Flush state and mark the engine as stopped."""
        self.shutdown_event.set()
        self.clear_session()

    def resume(self) -> None:
        """Clear the shutdown flag so the engine accepts work again."""
        self.shutdown_event.clear()

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def load_pinned_policy(self, path: str, device: str = "cuda", policy_type: str | None = None) -> None:
        """Load weights from ``path`` and keep that path for the life of the process.

        A later client ``RemotePolicyConfig`` can still update features, chunk length,
        device, and rename_map. Its model path is replaced with ``path``.
        """
        saved = _read_pretrained_config(path)
        resolved_type = policy_type or saved.get("type")
        if not resolved_type:
            raise ValueError(
                f"Could not read a policy type from {path}/config.json. Pass --policy-type."
            )
        actions_per_chunk = int(saved.get("n_action_steps") or 50)
        self._pinned_name_or_path = path
        self.load_policy(
            RemotePolicyConfig(
                policy_type=resolved_type,
                pretrained_name_or_path=path,
                lerobot_features={},
                actions_per_chunk=actions_per_chunk,
                device=device,
            )
        )

    def load_policy(self, config: RemotePolicyConfig) -> None:
        if self._pinned_name_or_path is not None:
            if config.pretrained_name_or_path != self._pinned_name_or_path:
                logger.info(
                    f"Client requested {config.pretrained_name_or_path}; "
                    f"keeping server model {self._pinned_name_or_path}"
                )
            config.pretrained_name_or_path = self._pinned_name_or_path
            if self.policy_type is not None and config.policy_type != self.policy_type:
                logger.info(
                    f"Client requested policy type {config.policy_type}; keeping {self.policy_type}"
                )
                config.policy_type = self.policy_type

        if config.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(f"Unsupported policy type {config.policy_type}. Supported: {SUPPORTED_POLICIES}")

        same_weights = (
            self.policy is not None
            and self.policy_type == config.policy_type
            and self._loaded_name_or_path == config.pretrained_name_or_path
        )
        if same_weights:
            self.device = config.device
            self.lerobot_features = config.lerobot_features
            self.actions_per_chunk = config.actions_per_chunk
            self.policy.to(self.device)
            if self._loaded_rename_map != config.rename_map:
                self._load_processors(config)
            logger.info(
                f"Policy already resident on {self.device}; skipped weight reload "
                f"({config.pretrained_name_or_path})"
            )
            return

        if self.policy is not None:
            logger.info("Replacing existing policy; unloading previous model first")
            self.unload_policy()

        self.device = config.device
        self.policy_type = config.policy_type
        self.lerobot_features = config.lerobot_features
        self.actions_per_chunk = config.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)
        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(config.pretrained_name_or_path)
        self.policy.to(self.device)
        self._loaded_name_or_path = config.pretrained_name_or_path
        if getattr(self.policy, "supports_rtc", lambda: False)():
            self.policy.config.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=int(os.environ.get("RTC_EXECUTION_HORIZON", "25")),
                max_guidance_weight=float(os.environ.get("RTC_MAX_GUIDANCE_WEIGHT", "10.0")),
                prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
            )
            self.policy.init_rtc_processor()
            logger.info(f"RTC ready: {self.policy.config.rtc_config}")
        self._load_processors(config)
        elapsed = time.perf_counter() - start
        logger.info(f"Policy loaded on {self.device} in {elapsed:.2f}s")

    def _load_processors(self, config: RemotePolicyConfig) -> None:
        device_override = {"device": self.device}
        preprocessor_overrides = {"device_processor": device_override}
        # Only override the rename_map if the client actually provides one.
        # This prevents wiping out a rename_map that might already be in the repo.
        if config.rename_map:
            preprocessor_overrides["rename_observations_processor"] = {"rename_map": config.rename_map}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=config.pretrained_name_or_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": device_override},
        )
        self._loaded_rename_map = dict(config.rename_map)

    # ------------------------------------------------------------------
    # Observation management
    # ------------------------------------------------------------------

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy."""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            logger.info(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        return True

    def enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation for inference.  Returns True if enqueued."""
        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            logger.info(f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}")

            if self.observation_queue.full():
                _ = self.observation_queue.get_nowait()
                logger.debug("Observation queue was full, removed oldest observation")

            self.observation_queue.put(obs)
            return True

        return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_action_chunk(
        self,
        observation: dict[str, torch.Tensor],
        inference_delay: int | None = None,
        prev_chunk_left_over: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get an action chunk from the policy.

        A leftover prefix is passed only when the client attached rtc_inference_delay.
        Without it, this is a plain predict_action_chunk call.
        """
        if prev_chunk_left_over is not None:
            chunk = self.policy.predict_action_chunk(
                observation,
                inference_delay=int(inference_delay),
                prev_chunk_left_over=prev_chunk_left_over,
            )
        else:
            chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances."""
        return [
            TimedAction(
                timestamp=t_0 + i * self.config.environment_dt,
                timestep=i_0 + i,
                action=action,
            )
            for i, action in enumerate(action_chunk)
        ]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Full inference pipeline: prepare → preprocess → infer → postprocess → time."""
        # 0. Decode any compressed (JPEG) images to numpy arrays
        raw_obs = _decode_compressed_images(observation_t.get_observation())

        # RTC is used only when the robot attaches rtc_inference_delay.
        # must_go means the robot has no actions left, so the previous plan is dropped.
        delay = raw_obs.pop("rtc_inference_delay", None)
        timestep = observation_t.get_timestep()
        prev = None
        if delay is not None and not observation_t.must_go and self._rtc_prev is not None:
            offset = timestep - self._rtc_prev_start
            if 0 <= offset < self._rtc_prev.shape[0]:
                prev = self._rtc_prev[offset:]

        # 1. Prepare observation
        start_prepare = time.perf_counter()
        observation: Observation = _raw_observation_to_observation(
            raw_obs,
            self.lerobot_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        # 2. Apply preprocessor
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        # 3. Get action chunk
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation, delay, prev)
        inference_time = time.perf_counter() - start_inference
        logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        # Save the normalized chunk before postprocess. RTC compares normalized values.
        self._rtc_prev = action_tensor[0].detach().clone()
        self._rtc_prev_start = timestep
        logger.info(
            f"RTC {'on' if prev is not None else 'off'} | delay={delay} | "
            f"prefix={0 if prev is None else prev.shape[0]} | inference={inference_time:.3f}s"
        )

        # 4. Apply postprocessor
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape
        processed_actions = []
        for i in range(chunk_size):
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        action_tensor = action_tensor.detach().cpu()

        # 5. Convert to TimedAction list
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )
        logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare: {1000 * prepare_time:.2f}ms | "
            f"Preprocess: {1000 * preprocessing_time:.2f}ms | "
            f"Inference: {1000 * inference_time:.2f}ms | "
            f"Postprocess: {1000 * postprocessing_time:.2f}ms | "
            f"Total: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def run_inference(self) -> list[TimedAction] | None:
        """Blocking call: wait for an observation, run inference, return actions.

        Returns ``None`` when the observation queue times out.
        """
        try:
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
        except Empty:
            return None

        logger.info(f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})")

        with self._predicted_timesteps_lock:
            self._predicted_timesteps.add(obs.get_timestep())

        return self._predict_action_chunk(obs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the engine to stop and release the loaded weights."""
        self.reset()
        self.unload_policy()
        logger.info("InferenceEngine stopping...")
