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

# Modified by Rintaro Kinoshita. gRPC-only CLI; model path can be pinned at startup.

"""Start the gRPC policy server.

Usage::

    python -m policy_server --host=0.0.0.0 --port=8080 --fps=30 \\
        --pretrained-name-or-path=outputs/pi05_so101_merged
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent import futures
from dataclasses import asdict
from pprint import pformat

from policy_server.inference_engine import InferenceEngine, InferenceEngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="gRPC policy server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--inference-latency",
        type=float,
        default=0.033,
        help="Target inference latency in seconds",
    )
    parser.add_argument(
        "--obs-queue-timeout",
        type=float,
        default=2.0,
        help="Observation queue timeout in seconds",
    )
    parser.add_argument(
        "--pretrained-name-or-path",
        default=None,
        help=(
            "Local model directory or Hub id to load before listening. "
            "When set, the model path sent by the client is ignored."
        ),
    )
    parser.add_argument(
        "--policy-type",
        default=None,
        help="Policy type. Read from config.json when omitted.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for the policy")
    args = parser.parse_args()

    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=log_level)

    import grpc

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.transport import services_pb2_grpc  # type: ignore

    from policy_server.grpc_server import PolicyServer

    config = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    logging.info(pformat(asdict(config)))

    engine = InferenceEngine(
        InferenceEngineConfig(
            fps=args.fps,
            inference_latency=args.inference_latency,
            obs_queue_timeout=args.obs_queue_timeout,
        )
    )
    if args.pretrained_name_or_path:
        logging.info(f"Loading policy from {args.pretrained_name_or_path} on {args.device}")
        engine.load_pinned_policy(
            args.pretrained_name_or_path,
            device=args.device,
            policy_type=args.policy_type,
        )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(PolicyServer(config, engine=engine), server)
    server.add_insecure_port(f"{args.host}:{args.port}")

    logging.info(f"gRPC PolicyServer started on {args.host}:{args.port}")
    server.start()
    server.wait_for_termination()
    logging.info("gRPC server terminated")


if __name__ == "__main__":
    main()
