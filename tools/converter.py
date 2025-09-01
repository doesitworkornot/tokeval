import json
import sys
from typing import Dict

import pandas as pd


def parquet_to_jsonl(parquet_path: str, jsonl_path: str) -> None:
    df = pd.read_parquet(parquet_path)
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for record in df.to_dict(orient="records"):
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def txt_to_jsonl(
    input_path: str, output_jsonl_path: str, label_map_path: Dict[str, int]
) -> None:
    label_set = set()
    data = []

    with open(input_path, "r", encoding="utf-8") as infile:
        tokens = []
        tags = []

        for line in infile:
            line = line.strip()
            if not line:
                if tokens:
                    data.append((tokens, tags))
                    tokens = []
                    tags = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                tag = parts[-1]
                tokens.append(token)
                tags.append(tag)
                label_set.add(tag)

        # Обработка последнего блока
        if tokens:
            data.append((tokens, tags))

    # Дополнение I- тэгов к B- тэгам, если их не было
    extended_labels = set(label_set)
    for label in label_set:
        if label.startswith("B-"):
            i_label = "I-" + label[2:]
            if i_label not in extended_labels:
                extended_labels.add(i_label)

    # Создание label2id (отсортированный для стабильности)
    label2id = {label: idx for idx, label in enumerate(sorted(extended_labels))}

    # Запись jsonl файла
    with open(output_jsonl_path, "w", encoding="utf-8") as out_jsonl:
        for tokens, tags in data:
            json_line = json.dumps(
                {"tokens": tokens, "ner_tags": [label2id[tag] for tag in tags]},
                ensure_ascii=False,
            )
            out_jsonl.write(json_line + "\n")

    # Сохранение label2id
    with open(label_map_path, "w", encoding="utf-8") as label_file:
        json.dump(label2id, label_file, ensure_ascii=False, indent=2)


label2id = {
    "B-ADJP": 0,
    "B-ADVP": 1,
    "B-CONJP": 2,
    "B-INTJ": 3,
    "B-LST": 4,
    "B-NP": 5,
    "B-PP": 6,
    "B-PRT": 7,
    "B-SBAR": 8,
    "B-UCP": 9,
    "B-VP": 10,
    "I-ADJP": 11,
    "I-ADVP": 12,
    "I-CONJP": 13,
    "I-INTJ": 14,
    "I-LST": 15,
    "I-NP": 16,
    "I-PP": 17,
    "I-PRT": 18,
    "I-SBAR": 19,
    "I-UCP": 20,
    "I-VP": 21,
    "O": 22,
}


def val_txt_to_jsonl(input_path: str, output_path: str) -> None:
    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        tokens = []
        tags = []

        for line in infile:
            line = line.strip()
            if not line:
                if tokens:  # конец блока
                    json_line = json.dumps(
                        {"tokens": tokens, "ner_tags": [label2id[tag] for tag in tags]},
                        ensure_ascii=False,
                    )
                    outfile.write(json_line + "\n")
                    tokens = []
                    tags = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                tag = parts[-1]
                tokens.append(token)
                tags.append(tag)

        # обработка последнего блока, если файл не заканчивается пустой строкой
        if tokens:
            json_line = json.dumps(
                {"tokens": tokens, "ner_tags": [label2id[tag] for tag in tags]},
                ensure_ascii=False,
            )
            outfile.write(json_line + "\n")


if __name__ == "__main__":
    # parquet_to_jsonl(sys.argv[1], sys.argv[2])
    # txt_to_jsonl(sys.argv[1], sys.argv[2], sys.argv[3])
    val_txt_to_jsonl(sys.argv[1], sys.argv[2])
