"""Main script for evaluating NER, RE, and Chunking classifiers.

Uses pre-computed embeddings and a specified model and tokenizer.
"""

import gc

from transformers import AutoModel, AutoTokenizer

from tokeval.embedding_evaluation.embeddings import NEREmbedder, REEmbedder
from tokeval.embedding_evaluation.validator import NERValidator, REValidator


def evaluate(model: AutoModel, tokenizer: AutoTokenizer) -> None:
    """Evaluate NER, RE, and Chunking classifiers using the provided model and tokenizer, and print the results.

    Args:
        model (AutoModel): Pre-trained model for generating embeddings.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the pre-trained model.

    """
    ner_embedder = NEREmbedder("./data/datasets/NER/multinerd/", model, tokenizer, cutoff=10000)
    ner = NERValidator(ner_embedder)
    ner.train()
    ner_f1, ner_acc = ner.get_results()
    del ner, ner_embedder
    gc.collect()

    re_embedder = REEmbedder("./data/datasets/RE/semeval2010_task8/", model, tokenizer, cutoff=10000)
    re = REValidator(re_embedder)
    re.train()
    re_f1, re_acc = re.get_results()
    del re
    gc.collect()

    chunk_embedder = NEREmbedder("./data/datasets/POS/conll2000/", model, tokenizer, cutoff=10000)
    chunk = NERValidator(chunk_embedder)
    chunk.train()
    chunk_f1, chunk_acc = chunk.get_results()
    del chunk
    gc.collect()

    print("\n\nNER")
    print(f"\nBest F1 Score: {ner_f1:.3f}")
    print(f"Best Accuracy: {ner_acc:.3f}")

    print("\n\nRelation Extraction")
    print(f"\nBest F1 Score: {re_f1:.3f}")
    print(f"Best Accuracy: {re_acc:.3f}")

    print("\n\nChunking")
    print(f"\nBest F1 Score: {chunk_f1:.3f}")
    print(f"Best Accuracy: {chunk_acc:.3f}")


if __name__ == "__main__":
    model_name = "gaunernst/bert-small-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModel.from_pretrained(model_name)
    evaluate(model, tokenizer)
