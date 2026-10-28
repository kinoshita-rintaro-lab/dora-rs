"""Upload a local LeRobot dataset to the Hugging Face Hub.

The repository is private unless --public is passed. The dataset id is
<logged-in-user>/so101 when --repo-id is omitted.

    uv run tools/push_dataset.py datasets/so101
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, whoami
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local LeRobot dataset to the Hugging Face Hub")
    parser.add_argument("root", type=Path, help="local LeRobot dataset directory, e.g. datasets/so101")
    parser.add_argument("--repo-id", default=None, help="Hub dataset id; default is <user>/so101")
    parser.add_argument("--public", action="store_true", help="create a public repository")
    args = parser.parse_args()
    root = args.root
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"{root} is not a LeRobot dataset (expected meta/info.json)")
    try:
        user = whoami()["name"]
    except Exception as error:
        raise SystemExit(f"not logged in to Hugging Face: {error}") from error
    repo_id = args.repo_id or f"{user}/so101"
    dataset = LeRobotDataset(repo_id, root=root)
    dataset.push_to_hub(private=not args.public)
    revision = HfApi().repo_info(repo_id, repo_type="dataset").sha
    visibility = "public" if args.public else "private"
    print(f"uploaded {repo_id} ({visibility})")
    print(f"revision {revision}")


if __name__ == "__main__":
    main()
