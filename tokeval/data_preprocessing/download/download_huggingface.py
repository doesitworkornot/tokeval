"""Module for downloading files from Hugging Face repositories."""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_file(repo_id: str, filename: str, local_dir: Path) -> None:
    """Download a single file from a Hugging Face dataset repository.

    Args:
        repo_id (str): The identifier of the Hugging Face dataset repository (e.g., "Babelscape/multinerd").
        filename (str): The path to the file within the repository to be downloaded (e.g., "train/train_en.jsonl").
        local_dir (Path): The local directory where the downloaded file should be saved.

    """
    local_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )


def download_files(repo_id: str, files: dict, local_dir: Path) -> None:
    """Download multiple files from a Hugging Face dataset repository.

    Args:
        repo_id (str): The identifier of the Hugging Face dataset repository (e.g., "Babelscape/multinerd").
        files (dict): A dictionary where keys are file identifiers (e.g., "train", "val")
            and values are the corresponding file paths in the repository.
        local_dir (Path): The local directory where the downloaded files should be saved.

    """
    for file in files.values():
        download_file(repo_id, file, local_dir)
    cache_path = local_dir / ".cache"
    shutil.rmtree(cache_path) if cache_path.exists() else None
