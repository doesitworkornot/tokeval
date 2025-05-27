from transformers import (
    AutoTokenizer,
    AutoModel,
)

from source.validator import NER_Validator
from source.validator import RE_Validator

def evaluate(model, tokenizer):
    ner = NER_Validator("./data/NER/multinerd/", model, tokenizer)
    ner_f1, ner_acc = ner.get_results()

    re = RE_Validator("./data/Relation_extraction/sem_eval_2010_task_8/", model, tokenizer)
    re_f1, re_acc = re.get_results()

    print('\n\nNER')
    print(f"\nBest F1 Score: {ner_f1:.4f}")
    print(f"Best Accuracy: {ner_acc:.4f}")

    
    print('\n\nRelation Extraction')
    print(f"\nBest F1 Score: {re_f1:.4f}")
    print(f"Best Accuracy: {re_acc:.4f}")


if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    evaluate(model, tokenizer)
