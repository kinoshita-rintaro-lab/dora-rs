# SO-101 Policy Client

Robot-side gRPC client for `policy_server`. It sends the policy loading
instructions and observations to the GPU machine, receives action
chunks, and passes them to `dora-so101-action-executor`.

## Inputs

- `observation`: the observer output, with `state`, `overhead`, `wrist`,
  and `task` fields.
- `queue_status`: `[queue_size, next_timestep]` from
  `dora-so101-action-executor`.

## Output

- `action_chunk`: a chunk of `float32[6]` actions. The first action's
  timestep is sent as the `timestep` metadata.

## Environment

- `SERVER_ADDRESS`: policy server address (`host:port`).
- `POLICY`: the merged model path on the server machine, or a Hub model
  id.
- `POLICY_TYPE`: `pi05`.
- `POLICY_DEVICE`: `cuda`.
- `ACTIONS_PER_CHUNK`: number of actions to request, `50`.
- `CHUNK_SIZE_THRESHOLD`: request more actions when the executor queue
  is at or below this fraction of a chunk, `0.5`.
- `TASK`: fallback task text when the observation has no task.
