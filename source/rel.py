from transformers import (
    AutoTokenizer,
    AutoModel,
)

from validator import RE_validator



if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    trainer = RE_validator("./data/Relation_extraction/sem_eval_2010_task_8/", model, tokenizer)