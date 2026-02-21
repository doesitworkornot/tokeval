"""Module defining paths used across the project."""

from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent.parent

DATA_FOLDER = PROJECT_FOLDER / "data"
DATASET_FOLDER = DATA_FOLDER / "datasets"
DATASET_FILE = DATA_FOLDER / "datasets.jsonl"
DATASET_LABELS = DATA_FOLDER / "dataset_labels"
LOG_DIR = PROJECT_FOLDER / "logs"
