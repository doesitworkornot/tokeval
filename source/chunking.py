from transformers import (
    AutoTokenizer,
    AutoModel,

)

from validator import NER_Validator


if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    trainer = NER_Validator("./data/POS_tagging/conll2000/", model, tokenizer)