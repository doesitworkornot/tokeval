"""Module for downloading datasets based on a configuration file."""

import json
import logging
from pathlib import Path

from tokeval.data_preprocessing.download.download_huggingface import download_files as download_files_hf
from tokeval.data_preprocessing.download.download_link import download_files as download_files_link
from tokeval.shared.log import get_logger, setup_logging
from tokeval.shared.paths import DATASET_FILE, DATASET_FOLDER

logger = get_logger(__name__)


def download_dataset(dataset: dict) -> None:
    """Load a dataset based on the provided dataset information.

    Args:
        dataset (dict): A dictionary containing dataset information,
            including type, name, link, and Hugging Face identifier.

    """
    logging.info(f"Processing dataset: {dataset['dataset']} of type {dataset['type']}...")
    dataset_path = DATASET_FOLDER / dataset["type"] / dataset["dataset"]

    hf_name = dataset.get("hugginface")
    if hf_name:
        data_files = dataset.get("files")
        branch = dataset.get("branch")
        download_files_hf(repo_id=hf_name, files=data_files, local_dir=dataset_path, branch=branch)
        return

    gh_link = dataset.get("github")
    if gh_link:
        data_files = dataset.get("files")
        download_files_link(url_link=gh_link, files=data_files, local_dir=dataset_path)
        return

    else:
        raise ValueError(f"Unsupported dataset type: {dataset['dataset']}. No valid download method found.")


def datasets_from_jsonl(datasets_path: Path) -> None:
    """Load all datasets specified in the datasets.jsonl file.

    Args:
        datasets_path (Path): Path to the datasets.jsonl file containing dataset information.

    """
    logging.info(f"Loading datasets from {datasets_path}...")
    with open(datasets_path, encoding="utf-8") as f:
        datasets = [json.loads(line) for line in f]

    for dataset in datasets:
        download_dataset(dataset)


if __name__ == "__main__":
    setup_logging(level="INFO")
    logger.info(f"Starting dataset loading process with {DATASET_FILE}...")
    datasets_from_jsonl(DATASET_FILE)
