# dora-so101-inference-ui

Web UI for remote SO-101 inference. Serves one control page and bridges
it to Dora: the page shows the `status` reported by
`dora-so101-policy-client`, and Start / Stop presses become `command`
outputs. The page never guesses the policy state on its own.

Nothing is inferred or executed until Start is pressed. Start carries
the task text in the `command` metadata, so the task shown on the page
is the task sent with every observation.

## Inputs and outputs

| Direction | Id | Content |
| --- | --- | --- |
| input | `tick` | wake-up timer, e.g. `dora/timer/hz/10` |
| input | `status` | policy client status, one JSON string |
| output | `command` | `start` or `stop` string; `start` has a `task` metadata entry |
| output | `torque` | bool. `false` until Start, `true` while running, `false` on Stop |

Wire `command` to `dora-so101-policy-client` and
`dora-so101-action-executor`: the client starts and stops requesting
inference, and the executor only releases actions between Start and
Stop (Stop also clears its queue). Wire `torque` to the follower.
Graphs that do not wire `torque` leave the arm holding and following
`cmd`.

## Environment

| Name | Default |
| --- | --- |
| `UI_HOST` | `127.0.0.1` |
| `UI_PORT` | `8080` |
| `TASK` | empty (page default) |

The page has no authentication. Keep `UI_HOST` at `127.0.0.1` unless
control from another machine on a trusted network is intended.
