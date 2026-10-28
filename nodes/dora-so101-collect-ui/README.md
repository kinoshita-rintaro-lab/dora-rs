# dora-so101-collect-ui

Web UI for `dora-so101-record`. Serves one control page and bridges it
to Dora: button presses become `command` outputs for the recorder, and
the recorder's `status` output drives what the page shows. The page
never guesses the recording state on its own.

## Inputs and outputs

| Direction | Id | Content |
| --- | --- | --- |
| input | `tick` | wake-up timer, e.g. `dora/timer/hz/10` |
| input | `status` | recorder status, one JSON string |
| output | `command` | recorder command string |

## Page

- Start begins an episode with the task text from the page.
- End finishes the episode. With "Ask success / fail after End"
  checked, the files are saved with the result pending and the page
  then asks for Success or Fail, which the recorder writes into
  `metadata.yaml`. With it unchecked, End saves as success at once.
  The checkbox starts from the `RESULT_PROMPT` environment variable
  and is remembered in the browser.
- Cancel discards the episode after a confirmation. During recording
  it drops the episode before it is saved. After End, while Success
  or Fail is still waiting, it deletes that saved episode.
- The header shows the recorder state, dataset path, per-stream
  counts, and the last recorder error, all from `status`.

There is no camera preview and no Quit button; stopping the dataflow
stops everything.

## Environment

| Name | Default |
| --- | --- |
| `UI_HOST` | `127.0.0.1` |
| `UI_PORT` | `8080` |
| `TASK` | empty (page default) |
| `RESULT_PROMPT` | `on` |

The page has no authentication. Keep `UI_HOST` at `127.0.0.1` unless
control from another machine on a trusted network is intended.
