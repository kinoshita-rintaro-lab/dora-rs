# dora-so101-record

Record SO-101 episodes as timestamped per-stream files. Adapted for
SO-101 from Enactic's OpenArm dataset recorder (Apache-2.0): each input
stream is collected independently with its own timestamps, cameras are
written as image files during the episode, and joint streams are
flushed to Parquet when the episode finishes. Streams are not
synchronized or resampled.

## Inputs

| Input id | Content |
| --- | --- |
| `command` | Episode control string, see below |
| `action` | `float32` joint vector, e.g. `leader/leader_cmd` |
| `state` | `float32` joint vector, e.g. `follower/follower_state` |
| `camera_<name>` | Encoded image, e.g. `camera_wrist: wrist-camera/image` |

`command` is a string array whose first element is one of `start`,
`success`, `fail`, `cancel`, `quit`. The `start` event may carry
`episode_number`, `task_index`, and `task` metadata; without
`episode_number` the next free number is used.

Timestamps come from each event's `timestamp` metadata (`datetime` is
converted to POSIX nanoseconds). `state` prefers
`observation_timestamp` when present. Events without a timestamp use
the receive time.

## Environment

| Name | Default |
| --- | --- |
| `DIRECTORY` | current directory |
| `NAME` | `dataset` |
| `METADATA_FILE` | none |
| `OPERATION_TYPE` | `teleop` |
| `ROBOT_TYPE` | `so101` |
| `JOINTS` | the six SO-101 joints |
| `TASK` | none |
| `AUTO_START` | unset |
| `AUTO_FINISH` | `success` |
| `TASK_INDEX` | `0` |

With `AUTO_START=true` an episode starts as soon as the node starts,
without a `command` input, and finishes when the node stops. This is
the mode used by `dataflow/episode-record.yaml`, where one dataflow
run records one episode. `AUTO_FINISH` (`success` or anything else
for failure) decides the success flag of that episode.

## Output

```text
{DIRECTORY}/{NAME}/
  metadata.yaml
  episodes/{number}/
    action/state.parquet   # timestamp, value: list<float32>
    obs/state.parquet      # timestamp, value: list<float32>
    cameras/{name}/{timestamp}.{encoding}
```

`metadata.yaml` holds the dataset format version, robot and joint
names, best-effort input frequencies, and one entry per finished
episode with its success flag, task, and per-stream frame counts.
Reusing a dataset directory resumes its episode list; recording an
episode number that already exists is an error.

## Example

```yaml
  - id: record
    path: dora-so101-record
    inputs:
      command: command/command
      action: leader/leader_cmd
      state: follower/follower_state
      camera_wrist: wrist-camera/image
      camera_overhead: overhead-camera/image
    env:
      DIRECTORY: /path/to/output
      NAME: so101
```
