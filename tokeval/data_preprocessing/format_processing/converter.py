"""Tools for converting data formats for token classification tasks."""

from pathlib import Path

import pandas as pd


def parquet_to_jsonl(parquet_path: Path, jsonl_path: Path) -> None:
    """Convert a parquet file to jsonl format.

    Args:
        parquet_path: Path to the input parquet file.
        jsonl_path: Path to the output jsonl file.

    """
    df = pd.read_parquet(parquet_path)
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)


def transform_jsonl_inplace(path: Path) -> None:
    """Transform a jsonl file to a common format for NER datasets.

    Args:
        path: path to the jsonl file to be transformed.

    """
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=["id", "pos_tags"], errors="ignore")
    df = df.rename(columns={"chunk_tags": "ner_tags"})
    df.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def process_datasets(dataset_folder: Path) -> None:
    """Process datasets in the given folder to a common format.

    Args:
        dataset_folder: Path to the folder containing the datasets to be processed.

    """
    for data_file in dataset_folder.rglob("*.parquet"):
        output_jsonl = data_file.with_suffix(".jsonl")
        parquet_to_jsonl(data_file, output_jsonl)
        data_file.unlink()
        if output_jsonl.parent.parent.name == "conll2000":
            transform_jsonl_inplace(output_jsonl)
