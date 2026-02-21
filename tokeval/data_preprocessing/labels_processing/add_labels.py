"""Script to add labels.json files to each dataset directory."""

import shutil

from tokeval.shared.log import get_logger, setup_logging
from tokeval.shared.paths import DATASET_FOLDER, DATASET_LABELS

setup_logging(level="INFO")
logger = get_logger(__name__)


def add_labels_to_datasets() -> None:
    """Add labels.json files to each dataset directory based on the dataset_labels folder."""
    for task_class in DATASET_FOLDER.iterdir():
        for dataset_dir in task_class.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.stem
                label_file = DATASET_LABELS / f"{dataset_name}.json"
                if not label_file.exists():
                    logger.warning(f"Label file for dataset {dataset_name} not found. Skipping.")
                    continue
                shutil.copy(label_file, dataset_dir / "labels.json")
                logger.info(f"Added labels.json to {dataset_dir}")


if __name__ == "__main__":
    add_labels_to_datasets()
