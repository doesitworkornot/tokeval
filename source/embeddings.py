import json
import os
from pathlib import Path
from datasets import Dataset, load_from_disk, concatenate_datasets
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForTokenClassification,

)


class Embedder:
    def __init__(self, dataset_path, model, tokenizer, cutoff=None):
        self.dataset_path = Path(dataset_path)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.hidden_size = 768
        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.cutoff = cutoff
        self.label2id = self.read_tags(self.dataset_path / 'labels.json')
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)

        self.train_set = self.load_dataset(self.dataset_path / 'train' / 'train.jsonl')
        self.val_set = self.load_dataset(self.dataset_path / 'val' / 'val.jsonl')


    def load_dataset(self, dataset_path):
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        if self.cutoff is not None:
            return Dataset.from_list(data[:self.cutoff])
        return Dataset.from_list(data)
    

    def get_embeddings(self):
        return self.vectorized_train, self.vectorized_val


    def read_tags(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.loads(file.read().strip())



class NER_Embedder(Embedder):
    def __init__(self, dataset_path, model, tokenizer, cutoff=1000):
        super().__init__(dataset_path, model, tokenizer, cutoff=cutoff)
        self.vectorized_train = self.flatten_dataset(self.vectorize(self.train_set))
        self.vectorized_val = self.flatten_dataset(self.vectorize(self.val_set))

    def process_example(self, example):
            flattened = []
            for emb, tag in zip(example["embeddings"], example["labels"]):
                flattened.append({
                    "embedding": np.array(emb),
                    "labels": tag
                })
            return flattened
    
    def flatten_dataset(self, original_dataset):
        new_data = []
        for example in tqdm(original_dataset, desc="Flattening"):
            new_data.extend(self.process_example(example))
        return Dataset.from_list(new_data)


    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs['labels'] = labels
        return tokenized_inputs


    def vectorize(self, dataset):
        tokenized_dataset = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names
        )
        collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            return_tensors="pt",
            padding=True
        )
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=8,
            collate_fn=collator,
            shuffle=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        temp_dir = tempfile.mkdtemp(prefix="vectorized_chunks_")
        chunk_size = 1000
        chunk_index = 0
        buffer = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Vectorizing"):
                labels = batch.pop("labels").cpu().numpy()
                input_lengths = batch["attention_mask"].sum(dim=1).tolist()
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = self.model(**batch)
                hidden_states = outputs.last_hidden_state.cpu().numpy()

                for i in range(hidden_states.shape[0]):
                    seq_len = input_lengths[i]
                    embedding = hidden_states[i, :seq_len]
                    label_seq = labels[i, :seq_len]

                    buffer.append({
                        "embeddings": embedding.tolist(),
                        "labels": label_seq.tolist()
                    })

                    if len(buffer) >= chunk_size:
                        chunk_dataset = Dataset.from_list(buffer)
                        chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index}")
                        chunk_dataset.save_to_disk(chunk_path)
                        buffer = []
                        chunk_index += 1

            if buffer:
                chunk_dataset = Dataset.from_list(buffer)
                chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index}")
                chunk_dataset.save_to_disk(chunk_path)
                chunk_index += 1

        all_chunks = []
        for i in range(chunk_index):
            chunk_path = os.path.join(temp_dir, f"chunk_{i}")
            all_chunks.append(load_from_disk(chunk_path))

        self.hidden_size = embedding.shape[1]
        full_dataset = concatenate_datasets(all_chunks)
        return full_dataset



class RE_Embedder(Embedder):
    def __init__(self, dataset_path, model, tokenizer):
        super().__init__(dataset_path, model, tokenizer)
        self.vectorized_train = self.vectorize(self.train_set)
        self.vectorized_val = self.vectorize(self.val_set)
        

    def preprocess_sentence(self, sentence):
        e1_start_tag = sentence.index("<e1>")
        e1_end_tag = sentence.index("</e1>")
        e2_start_tag = sentence.index("<e2>")
        e2_end_tag = sentence.index("</e2>")

        e1_text = sentence[e1_start_tag + 4 : e1_end_tag]
        e2_text = sentence[e2_start_tag + 4 : e2_end_tag]

   
        clean_sentence = (
            sentence.replace("<e1>", "")
                    .replace("</e1>", "")
                    .replace("<e2>", "")
                    .replace("</e2>", "")
        )

        e1_clean_start = clean_sentence.index(e1_text)
        e2_clean_start = clean_sentence.index(e2_text)

        return clean_sentence, {
            "e1": (e1_clean_start, e1_clean_start + len(e1_text)),
            "e2": (e2_clean_start, e2_clean_start + len(e2_text))
        }


    def vectorize(self, dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        processed_data = []
        for item in tqdm(dataset, desc="Vectorizing"):
            sentence = item["sentence"]
            relation = item["relation"]

            clean_sentence, entity_positions = self.preprocess_sentence(sentence)

            encoding = self.tokenizer(clean_sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True)
            offsets = encoding["offset_mapping"][0].tolist()
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            def find_token_index(char_pos):
                """Находит индекс токена, начинающегося в позиции char_pos"""
                for i, (start, end) in enumerate(offsets):
                    if start == char_pos:
                        return i
                return None

            e1_token_idx = find_token_index(entity_positions["e1"][0])
            e2_token_idx = find_token_index(entity_positions["e2"][0])

            if e1_token_idx is None or e2_token_idx is None:
                continue  # Пропускаем, если не удалось найти

            e1_emb = last_hidden[e1_token_idx].tolist()
            e2_emb = last_hidden[e2_token_idx].tolist()

            processed_data.append({
                "e1_embedding": e1_emb,
                "e2_embedding": e2_emb,
                "label": relation
            })
        self.hidden_size = len(e1_emb)
        dataset = Dataset.from_list(processed_data)
        return dataset