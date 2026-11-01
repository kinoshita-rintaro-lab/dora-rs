# Async Action Executor

Queues an action chunk and sends one action on each tick. Sending the
next action does not wait for a new chunk to arrive.

## Inputs

- `tick`: the control-rate timer.
- `command`: `start` or `stop` from `dora-so101-inference-ui`. Actions
  are only released between `start` and `stop`; `stop` also clears the
  queue. The executor starts stopped.
- `action_chunk`: a chunk of actions. Each action has 6 joint values.
  A flat `float32` array whose length is a multiple of 6 is also accepted.
  The `timestep` metadata, when present, is the first action's timestep.

`INFERENCE_MODE` selects how a new chunk is applied. It must match
`dora-so101-policy-client`.

- `async` (default): a new chunk replaces actions that have not been
  executed yet. Actions whose timestep has already passed are dropped.
  The new chunk is shifted so its first executed action matches the
  action currently commanded, and the first `BLEND_STEPS` actions
  (default 10, about 0.33 s at 30 Hz) are then cross-faded from the
  actions they replace.
- `rtc`: same replacement and dropping, without the shift or the
  cross-fade. The policy server has already continued the previous chunk.
- `sync`: a chunk is executed as the model produced it. A chunk that
  arrives while one is still playing is ignored.

## Outputs

- `action`: one `float32[6]` action, in the same order as the follower
  command input.
- `queue_status`: `[queue_size, next_timestep]` as `int64[2]`, sent
  whenever the queue changes.

## Wiring

```yaml
  - id: dora-so101-action-executor
    path: dora-so101-action-executor
    inputs:
      tick: dora/timer/hz/30
      action_chunk: policy-client/action_chunk
    outputs:
      - action
      - queue_status
```

The follower can consume the output directly:

```yaml
      cmd: dora-so101-action-executor/action
```
