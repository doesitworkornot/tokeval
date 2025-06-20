import math
import pathlib

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
import numpy as np

from source.embeddings import NER_Embedder, RE_Embedder
from source.classifier import Classifier



class Validator():
    def __init__(self, hs, ds_len, id2label, nc):
        self.id2label = id2label
        self.classifier = Classifier(hs, nc)
        self.best_f1, self.best_acc = 0.0, 0.0  
        self.ds_len = ds_len


    def get_results(self):
        return self.best_f1, self.best_acc      


    def compute_metrics(self, predictions, labels):
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(predictions, labels) if l != -100]
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(predictions, labels) if l != -100]
        ]

        results = classification_report(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
        acc = accuracy_score(true_labels, true_predictions)
        print(results)
        if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_acc = acc
        return {
            'f1': f1,
            'accuracy': acc,
        }
    

    def train(self):
        bs = 128
        num_train_epochs = 5
        train_len = self.ds_len
        steps_per_epoch = math.ceil(train_len / bs)
        max_steps = steps_per_epoch * num_train_epochs

        print(f"Max steps: {max_steps}, Batch size: {bs}, Dataset length: {train_len}")

        train_loader = DataLoader(self.vectorized_train, batch_size=bs, collate_fn=self.collate_fn)
        val_loader = DataLoader(self.vectorized_val, batch_size=bs*2, collate_fn=self.collate_fn)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=5e-5, weight_decay=1e-4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

        step = 0
        for epoch in range(num_train_epochs):
            self.classifier.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_train_epochs}"):
                step += 1
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = self.classifier(**inputs)
                loss = criterion(outputs['logits'], labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if step % steps_per_epoch == 0:
                    self.classifier.eval()
                    all_preds, all_labels = [], []

                    with torch.no_grad():
                        for val_batch in val_loader:
                            inputs = {k: v.to(device) for k, v in val_batch.items() if k != "labels"}
                            labels = val_batch["labels"]
                            outputs = self.classifier(**inputs)
                            logits = outputs['logits']
                            preds = torch.argmax(logits, dim=-1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())

                    metrics = self.compute_metrics(all_preds, all_labels)
                    print(f"[Step {step}] Eval metrics: {metrics}")

            avg_train_loss = train_loss / steps_per_epoch
            print(f"Epoch {epoch+1} completed. Avg train loss: {avg_train_loss:.4f}")

        train_ds = pathlib.Path('./vectorized_train.parquet')
        val_ds = pathlib.Path('./vectorized_val.parquet')
        if train_ds.is_file():
            pathlib.Path.unlink(train_ds)
        if val_ds.is_file():
            pathlib.Path.unlink(val_ds)



class NER_Validator(Validator, NER_Embedder):
    def __init__(self, dataset_path, model, tokenizer, cutoff=1000):
        NER_Embedder.__init__(self, dataset_path, model, tokenizer, cutoff=cutoff)
        Validator.__init__(self, self.hidden_size, self.ds_len, id2label=self.id2label, nc=self.num_classes)
        self.vectorized_train, self.vectorized_val = self.get_embeddings()
        self.train()


    def collate_fn(self, batch):
        return {
            'embeddings': torch.stack([(f['embedding']).clone().detach() for f in batch]),
            'labels': torch.tensor([f['labels'] for f in batch])
        }



class RE_Validator(Validator, RE_Embedder):
    def __init__(self, dataset_path, model, tokenizer, cutoff=1000):
        RE_Embedder.__init__(self, dataset_path, model, tokenizer, cutoff=cutoff)
        Validator.__init__(self, self.hidden_size * 2, self.ds_len, id2label=self.id2label, nc=self.num_classes)
        self.vectorized_train, self.vectorized_val = self.get_embeddings()
        self.train()
     

    def collate_fn(self, batch):
        e1_embeddings = torch.stack([f['e1_embedding'] for f in batch])
        e2_embeddings = torch.stack([f['e2_embedding'] for f in batch])
        combined_embeddings = torch.cat([e1_embeddings, e2_embeddings], dim=-1)

        labels = torch.tensor([f['label'] for f in batch])

        return {
            'embeddings': combined_embeddings,
            'labels': labels
        }



