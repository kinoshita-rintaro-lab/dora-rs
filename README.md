# dora-so101

Run all commands from the repository root.

## Record

```bash
dora run dataflow/episode-record.yaml
```

Open http://127.0.0.1:8080. Start begins an episode, End saves it, then mark it Success or Fail. Cancel discards it. Unmarked episodes are skipped at conversion. Episodes are saved to `record/pick-and-place`.

## Remote inference

Put the policy server address in `.env` at the repository root (not committed):

```bash
SERVER_ADDRESS=<host>:<port>
```

```bash
dora run dataflow/remote_inference.yaml
```

Open http://127.0.0.1:8081. A `SERVER_ADDRESS` set in the shell overrides `.env`.

## Check synchronization

Prints per-episode counts without writing files. `--slop` is the matching window in seconds around each reference frame (default 0.020). Episodes dropping at least `--max-drop` of reference frames (default 0.05) are marked `reject`.

```bash
uv run tools/curation.py record/pick-and-place --reference overhead --slop 0.020
```

## Convert to LeRobot

Use the same `--slop`. Leader commands become `action`; follower state becomes `observation.state`.

```bash
uv run tools/record2lerobot.py record/pick-and-place --root datasets/so101-pick-and-place --reference overhead --slop 0.020
```

Add `--success-only` to keep only successful episodes. Rejected episodes are skipped. Re-running on the same `--root` skips episodes already in `meta/conversion.json` and refuses a different fps or slop.

## Upload to the Hub

Log in to Hugging Face first. The dataset is private and named `<user>/so101` by default.

```bash
uv run tools/push_dataset.py datasets/so101-pick-and-place
```

Use `--repo-id <user>/<name>` for another repository and `--public` to make it public.
