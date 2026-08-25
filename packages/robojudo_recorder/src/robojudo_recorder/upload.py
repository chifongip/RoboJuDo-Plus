"""Explicit, post-capture upload of a local LeRobot dataset to Hugging Face."""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a completed RoboJuDo LeRobot v3 dataset")
    parser.add_argument("dataset", type=Path, help="Local dataset root, for example record_data/x2_upper_body")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset id, for example user/x2_upper_body")
    parser.add_argument("--private", action="store_true", help="Create the dataset repository as private")
    parser.add_argument(
        "--commit-message",
        default="Upload RoboJuDo LeRobot v3 dataset",
        help="Commit message used for the upload",
    )
    return parser.parse_args()


def upload_dataset(dataset: Path, repo_id: str, *, private: bool, commit_message: str) -> None:
    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError(f"dataset directory does not exist: {dataset}")
    if not repo_id.strip() or "/" not in repo_id:
        raise ValueError("--repo-id must have the form <user-or-org>/<dataset>")
    if not commit_message.strip():
        raise ValueError("--commit-message must not be empty")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("install the optional dependency first: pip install -e '.[hub]'") from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dataset),
        commit_message=commit_message,
    )
    logger.info("Uploaded %s to https://huggingface.co/datasets/%s", dataset, repo_id)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    upload_dataset(args.dataset, args.repo_id, private=args.private, commit_message=args.commit_message)


if __name__ == "__main__":
    main()
