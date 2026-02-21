"""Main pipeline for building dataset from predetermined opensource links."""

from tokeval.data_preprocessing.download.download_datasets import datasets_from_jsonl
from tokeval.data_preprocessing.labels_processing.add_labels import add_labels_to_datasets
from tokeval.shared.log import get_logger, setup_logging
from tokeval.shared.paths import DATASET_FILE


def build_datasets() -> None:
    """Build datasets from specifications in the datasets.jsonl."""
    logger.info("Starting dataset downloading process...")
    datasets_from_jsonl(DATASET_FILE)
    logger.info("Starting enriching datasets with labels...")
    add_labels_to_datasets()


if __name__ == "__main__":
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    logger.info("Application started")
    build_datasets()
