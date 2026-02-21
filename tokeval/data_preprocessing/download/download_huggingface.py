"""Module for downloading files from Hugging Face repositories."""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from tokeval.shared.log import get_logger

logger = get_logger(__name__)


def download_files(repo_id: str, files: dict, local_dir: Path, branch: str | None) -> None:
    """Download files and rename them using dict keys as folder and filename.

    Args:
        repo_id (str): The identifier of the Hugging Face dataset repository
            (e.g., "Babelscape/multinerd").
        files (dict): A dictionary where keys are file identifiers (e.g., "train", "val")
            and values are the corresponding file paths in the repository.
        local_dir (Path): The local directory where the downloaded files should be saved.
        branch (str | None): The branch or revision to download from the repository.

    Example:
        files = {
            "train": "default/train/0000.parquet",
            "val": "default/test/0000.parquet"
        }

    Result:
        local_dir/
            train/train.parquet
            val/val.parquet

    """
    local_dir.mkdir(parents=True, exist_ok=True)

    for key, remote_path in files.items():
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=branch,
        )

        downloaded_path = Path(downloaded_path)
        ext = downloaded_path.suffix

        target_dir = local_dir / key
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / f"{key}{ext}"

        shutil.copy2(downloaded_path, target_file)

        logger.info(f"Downloaded {remote_path} → {target_file}")

    cache_path = local_dir / ".cache"
    if cache_path.exists():
        shutil.rmtree(cache_path)
