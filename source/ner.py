import json
import os
from pathlib import Path
from datasets import Dataset, load_from_disk, concatenate_datasets
import tempfile
import shutil

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


class NER_classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=32):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.silu = nn.SiLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels=None):
        # Прямое распространение
        x = self.batch_norm1(embeddings)
        x = self.linear1(x)
        x = self.batch_norm2(x)
        x = self.silu(x)
        logits = self.classifier(x)
        
        # Вычисление потерь
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}



class NER_validator:
    def __init__(self, dataset_path, model, tokenizer):
        self.dataset_path = Path(dataset_path)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.hidden_size = 768
        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.label2id = self.read_tags(self.dataset_path / 'ner_tags.json')
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)

        self.train_set = self.load_dataset(self.dataset_path / 'train' / 'train_en.jsonl')
        self.val_set = self.load_dataset(self.dataset_path / 'val' / 'val_en.jsonl')

        self.vectorized_train = self.flatten_dataset(self.vectorize(self.train_set))
        self.vectorized_val = self.flatten_dataset(self.vectorize(self.val_set))

        self.classifier = NER_classifier(self.hidden_size, self.num_classes)
        self.train()
        
    
    def get_results(self):
        return self.best_f1, self.best_acc      


    def load_dataset(self, dataset_path):
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return Dataset.from_list(data[:100])


    def read_tags(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.loads(file.read().strip())


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
                        "ner_tags": label_seq.tolist()
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

        all_chunks = []
        for i in range(chunk_index + 1):
            chunk_path = os.path.join(temp_dir, f"chunk_{i}")
            all_chunks.append(load_from_disk(chunk_path))

        self.hidden_size = embedding.shape[1]
        full_dataset = concatenate_datasets(all_chunks)
        return full_dataset


    def process_example(self, example):
            flattened = []
            for emb, tag in zip(example["embeddings"], example["ner_tags"]):
                flattened.append({
                    "embedding": np.array(emb),
                    "ner_tag": tag
                })
            return flattened
    
    def flatten_dataset(self, original_dataset):
        new_data = []
        for example in tqdm(original_dataset, desc="Flattening"):
            new_data.extend(self.process_example(example))
        return Dataset.from_list(new_data)

    def collate_fn(self, batch):
            return {
                'embeddings': torch.stack([torch.tensor(f['embedding']) for f in batch]),
                'labels': torch.tensor([f['ner_tag'] for f in batch])
            }
    def compute_metrics(self, pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
  
            current_f1 = f1_score(labels, preds, average='weighted')
            current_acc = accuracy_score(labels, preds)
  
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_acc = current_acc

            return {
                'f1': current_f1,
                'accuracy': current_acc
            }
    
    def train(self):
        training_args = TrainingArguments(
            num_train_epochs=10,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            eval_strategy='epoch',  # Валидация после каждой эпохи
            logging_strategy='epoch',     # Логирование после каждой эпохи
            remove_unused_columns=False,
            output_dir='./tmp',
            save_steps=0,
            load_best_model_at_end=False  # Не загружать лучшую модель
        )

        trainer = Trainer(
            model=self.classifier,
            args=training_args,
            train_dataset=self.vectorized_train,
            eval_dataset=self.vectorized_val,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics  # Добавляем вычисление метрик
        )

        trainer.train()

        if os.path.exists('./tmp'):
            shutil.rmtree('./tmp')
        if os.path.exists('./trainer_output'):
            shutil.rmtree('./trainer_output')




if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    trainer = NER_validator("./data/NER/multinerd/", model, tokenizer)