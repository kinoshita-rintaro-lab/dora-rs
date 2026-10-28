# Async Action Executor

Queues an action chunk and sends one action on each tick. Sending the
next action does not wait for a new chunk to arrive.

## Inputs

- `tick`: the control-rate timer.
- `action_chunk`: a chunk of actions. Each action has 6 joint values.
  A flat `float32` array whose length is a multiple of 6 is also accepted.
  The `timestep` metadata, when present, is the first action's timestep.

A new chunk replaces actions that have not been executed yet. Actions
whose timestep has already passed are dropped.

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
