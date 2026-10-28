# SO-101 Policy Client

Robot-side gRPC client for `policy_server`. It sends the policy loading
instructions and observations to the GPU machine, receives action
chunks, and passes them to `dora-so101-action-executor`.

## Inputs

- `observation`: the observer output, with `state`, `overhead`, `wrist`,
  and `task` fields.
- `queue_status`: `[queue_size, next_timestep]` from
  `dora-so101-action-executor`.
- `command`: `start` or `stop` from `dora-so101-inference-ui`. `start`
  carries the task text in its `task` metadata.

## Outputs

- `action_chunk`: a chunk of `float32[6]` actions. The first action's
  timestep is sent as the `timestep` metadata.
- `status`: one JSON string for the inference UI, with `state`
  (`connecting`, `ready`, `running`, or `error`), the active task, the
  executor queue size, and the last error.

The client connects and loads the policy at startup, then waits.
No observation is sent to the server until `start` arrives, and the
task from `start` replaces the task in every observation while running.
`stop` drops any inference result that is still in flight.

## Environment

- `SERVER_ADDRESS`: policy server address (`host:port`).
- `POLICY`: optional override for the merged model path on the server
  machine, or a Hub model id. When unset, the client asks the server to
  load `outputs/pi05_so101_merged`, relative to the server's working
  directory.
- `POLICY_TYPE`: `pi05`.
- `POLICY_DEVICE`: `cuda`.
- `ACTIONS_PER_CHUNK`: number of actions to request, `50`.
- `CHUNK_SIZE_THRESHOLD`: request more actions when the executor queue
  is at or below this fraction of a chunk, `0.5`.
- `TASK`: fallback task text when the observation has no task.
