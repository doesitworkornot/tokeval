"""Main script for evaluating token classification models on NER, Relation Extraction, and Chunking tasks."""

import gc

import pandas as pd
from transformers import AutoModel, AutoTokenizer

from tokeval.embedding_evaluation.embeddings import NEREmbedder, REEmbedder
from tokeval.embedding_evaluation.validator import NERValidator, REValidator

TASKS = {
    "NER": {
        "dataset_path": "./data/datasets/NER/multinerd/",
        "embedder_class": NEREmbedder,
        "validator_class": NERValidator,
    },
    "Relation Extraction": {
        "dataset_path": "./data/datasets/RE/semeval2010_task8/",
        "embedder_class": REEmbedder,
        "validator_class": REValidator,
    },
    "Chunking": {
        "dataset_path": "./data/datasets/POS/conll2000/",
        "embedder_class": NEREmbedder,
        "validator_class": NERValidator,
    },
}


def evaluate_model(model_name: str, cutoff: int = 10000) -> dict:
    """Evaluate all tasks for a given model and return results as a dictionary."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    results = {"model": model_name}

    for task_name, task_info in TASKS.items():
        embedder = task_info["embedder_class"](task_info["dataset_path"], model, tokenizer, cutoff=cutoff)
        validator = task_info["validator_class"](embedder)
        validator.train()
        f1, f1_low, f1_high, acc = validator.get_results()
        results[f"{task_name} F1"] = f1
        results[f"{task_name} F1_CI95_LOW"] = f1_low
        results[f"{task_name} F1_CI95_HIGH"] = f1_high
        results[f"{task_name} Accuracy"] = acc

        del embedder, validator
        gc.collect()

    del model, tokenizer
    gc.collect()
    return results


if __name__ == "__main__":
    model_names = [
        "Qwen/Qwen3.5-0.8B",
    ]

    all_results = []
    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(model_name)
        all_results.append(result)

    df = pd.DataFrame(all_results)
    print("\n\nBenchmark Results:")
    print(df)

    df.to_csv("benchmark_results2.csv", index=False)
