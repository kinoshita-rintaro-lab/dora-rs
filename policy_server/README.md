# Policy Server

GPU-side gRPC server for SO-101. It loads a LeRobot policy and returns
action chunks for observations sent by the robot-side client.

The policy is not loaded when the server starts. It is loaded when the
client sends its policy instructions.

## Prerequisites

Install LeRobot before installing this package:

```bash
pip install lerobot "lerobot[async,pi,peft]"
```

## Install

Run this in the GPU machine's Python environment:

```bash
uv pip install -e ./policy_server
```

## Usage

```bash
policy-server --host=0.0.0.0 --port=8080 --fps=30
```

The same entry point is available as a module:

```bash
python -m policy_server --host=0.0.0.0 --port=8080 --fps=30
```

## Files

- `server.py` parses the command-line options and starts the gRPC server.
- `grpc_server.py` receives policy instructions, observations, and action
  requests.
- `inference_engine.py` loads the policy, prepares each observation, and
  predicts an action chunk.
