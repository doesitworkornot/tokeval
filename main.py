from transformers import (
    AutoTokenizer,
    AutoModel,
)

from source.ner import NER_validator

def evaluate(model, tokenizer):
    ner = NER_validator("./data/NER/multinerd/", model, tokenizer)
    f1, acc = ner.get_results()
    print('\nNER')
    print(f"\nBest F1 Score: {f1:.4f}")
    print(f"Best Accuracy: {acc:.4f}")


if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    evaluate(model, tokenizer)
