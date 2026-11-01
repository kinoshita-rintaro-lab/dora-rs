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

- `SERVER_ADDRESS`: policy server address (`host:port`). Read from the repo-root `.env` when this is empty. A non-empty environment variable wins.
- `POLICY`: optional override for the merged model path on the server
  machine, or a Hub model id. When unset, the client asks the server to
  load `outputs/pi05_so101_merged`, relative to the server's working
  directory.
- `POLICY_TYPE`: `pi05`.
- `POLICY_DEVICE`: `cuda`.
- `ACTIONS_PER_CHUNK`: number of actions to request, `50`.
- `INFERENCE_MODE`: `sync`, `async`, or `rtc`. `sync` sends the next
  observation only after the executor queue has drained, and does not
  replace a chunk that is still playing. `async` overlaps inference
  with execution. `rtc` does the same and adds `rtc_inference_delay`
  (the previous round trip, in control steps) to each observation, which
  the RTC policy server uses to continue the previous chunk. Must match
  the executor. Default `async`.
- `CHUNK_SIZE_THRESHOLD`: in `async` and `rtc`, request more actions when
  the executor queue is at or below this fraction of a chunk, `0.5`.
  `rtc` wants about `0.6`, so the previous chunk still covers the
  server's execution horizon.
- `RTC_INFERENCE_DELAY`: steps assumed for the first request, before a
  round trip has been measured, `12`.
- `CONTROL_FPS`: control rate used to turn the round trip into steps, `30`.
- `TASK`: fallback task text when the observation has no task.
