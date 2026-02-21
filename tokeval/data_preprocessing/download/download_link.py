"""Download files from URL links and save them to a local directory as dataset."""

import shutil
from pathlib import Path

import requests


def download_files(url_link: str, files: dict, local_dir: Path) -> None:
    """Download multiple files from a Hugging Face dataset repository.

    Args:
        url_link (str): The URL link to the dataset repository.
        files (dict): A dictionary where keys are file identifiers (e.g., "train", "val")
            and values are the corresponding file paths in the repository.
        local_dir (Path): The local directory where the downloaded files should be saved.

    """
    local_dir.mkdir(parents=True, exist_ok=True)
    for key, remote_path in files.items():
        url = f"{url_link.rstrip('/')}/{remote_path}"
        ext = Path(remote_path).suffix

        target_dir = local_dir / key
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / f"{key}{ext}"

        response = requests.get(url, stream=True, timeout=100)
        response.raise_for_status()

        with open(target_file, "wb") as f:
            shutil.copyfileobj(response.raw, f)
