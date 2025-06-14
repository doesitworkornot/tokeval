import os
import shutil

import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    Trainer,
    TrainingArguments,
)

from source.embeddings import NER_Embedder, RE_Embedder
from source.classifier import Classifier



class Validator():
    def __init__(self, hs):
        self.classifier = Classifier(hs, self.num_classes)
        self.best_f1, self.best_acc = 0.0, 0.0  


    def get_results(self):
        return self.best_f1, self.best_acc      


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
            num_train_epochs=5,
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



class NER_Validator(Validator, NER_Embedder):
    def __init__(self, dataset_path, model, tokenizer, cutoff=1000):
        NER_Embedder.__init__(self, dataset_path, model, tokenizer, cutoff=cutoff)
        Validator.__init__(self, self.hidden_size)
        self.vectorized_train, self.vectorized_val = self.get_embeddings()
        self.train()


    def collate_fn(self, batch):
        return {
            'embeddings': torch.stack([torch.tensor(f['embedding']) for f in batch]),
            'labels': torch.tensor([f['labels'] for f in batch])
        }



class RE_Validator(Validator, RE_Embedder):
    def __init__(self, dataset_path, model, tokenizer):
        RE_Embedder.__init__(self, dataset_path, model, tokenizer)
        Validator.__init__(self, self.hidden_size * 2)
        self.vectorized_train, self.vectorized_val = self.get_embeddings()
        self.train()
     

    def collate_fn(self, batch):
        e1_embeddings = torch.stack([torch.tensor(f['e1_embedding']) for f in batch])
        e2_embeddings = torch.stack([torch.tensor(f['e2_embedding']) for f in batch])
        combined_embeddings = torch.cat([e1_embeddings, e2_embeddings], dim=-1)

        labels = torch.tensor([f['label'] for f in batch], dtype=torch.long)

        return {
            'embeddings': combined_embeddings,  # объединённые эмбеддинги
            'labels': labels
        }



