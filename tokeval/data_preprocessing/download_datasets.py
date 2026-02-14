"""Module for downloading datasets based on a configuration file."""

import json
import logging
from pathlib import Path

from tokeval.data_preprocessing.download.download_huggingface import download_files
from tokeval.shared.log import get_logger, setup_logging
from tokeval.shared.paths import DATASET_FILE, DATASET_FOLDER

setup_logging(level="INFO")

logger = get_logger(__name__)

logger.info("Application started")


def download_dataset(dataset: dict) -> None:
    """Load a dataset based on the provided dataset information.

    Args:
        dataset (dict): A dictionary containing dataset information,
            including type, name, link, and Hugging Face identifier.

    """
    logging.info(f"Processing dataset: {dataset['dataset']} of type {dataset['type']}...")
    if dataset["type"] == "NER":
        hf_name = dataset.get("hugginface")
        logging.info(f"Loading dataset from Hugging Face: {hf_name}...")
        dataset_path = DATASET_FOLDER / dataset["type"] / dataset["dataset"]
        data_files = dataset.get("files")
        download_files(repo_id=hf_name, files=data_files, local_dir=dataset_path)

    elif dataset["type"] == "POS":
        # Implement loading logic for POS datasets
        pass
    elif dataset["type"] == "RE":
        # Implement loading logic for RE datasets
        pass
    else:
        raise ValueError(f"Unsupported dataset type: {dataset['type']}")


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
    logger.info(f"Starting dataset loading process with {DATASET_FILE}...")
    print("hello")
    datasets_from_jsonl(DATASET_FILE)
