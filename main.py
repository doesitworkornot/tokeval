from transformers import (
    AutoTokenizer,
    AutoModel,
)
import gc

from source.validator import NER_Validator
from source.validator import RE_Validator

def evaluate(model, tokenizer):
    ner = NER_Validator("./data/NER/multinerd/", model, tokenizer, cutoff=None)
    ner_f1, ner_acc = ner.get_results()
    del ner
    gc.collect()

    re = RE_Validator("./data/Relation_extraction/sem_eval_2010_task_8/", model, tokenizer, cutoff=None)
    re_f1, re_acc = re.get_results()
    del re
    gc.collect()

    chunk = NER_Validator("./data/POS_tagging/conll2000", model, tokenizer, cutoff=None)
    chunk_f1, chunk_acc = chunk.get_results()
    del chunk
    gc.collect()

    print('\n\nNER')
    print(f"\nBest F1 Score: {ner_f1:.3f}")
    print(f"Best Accuracy: {ner_acc:.3f}")
    
    print('\n\nRelation Extraction')
    print(f"\nBest F1 Score: {re_f1:.3f}")
    print(f"Best Accuracy: {re_acc:.3f}")

    print('\n\nChunking')
    print(f"\nBest F1 Score: {chunk_f1:.3f}")
    print(f"Best Accuracy: {chunk_acc:.3f}")





if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    model = AutoModel.from_pretrained(model_name)
    evaluate(model, tokenizer)
