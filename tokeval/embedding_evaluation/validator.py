"""Validator module for training and evaluating NER and RE classifiers using pre-computed embeddings."""

import math
import pathlib
from typing import Any

import torch
from seqeval.metrics import accuracy_score, classification_report, f1_score
from source.classifier import Classifier
from source.embeddings import NEREmbedder, REEmbedder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class Validator:
    """Validator class for training and evaluating classifiers on NER and RE tasks."""

    def __init__(self, hs: int, ds_len: int, id2label: dict[int, str], nc: int) -> None:
        """Initialize the Validator with hidden size, dataset length, label mapping, and number of classes.

        Args:
            hs (int): Hidden size of the embeddings.
            ds_len (int): Length of the dataset.
            id2label (dict[int, str]): Mapping from label IDs to label names.
            nc (int): Number of classes for classification.

        """
        self.id2label = id2label
        self.classifier = Classifier(hs, nc)
        self.best_f1, self.best_acc = 0.0, 0.0
        self.ds_len = ds_len

    def get_results(self) -> tuple[float, float]:
        """Return the best F1 score and accuracy achieved during validation.

        Returns:
            tuple[float, float]: Best F1 score and accuracy.

        """
        return self.best_f1, self.best_acc

    def compute_metrics(
        self,
        predictions: list[int],
        labels: list[int],
    ) -> dict[str, float]:
        """Compute F1 score and accuracy based on predictions and true labels.

        Args:
            predictions (list[int]): List of predicted label IDs.
            labels (list[int]): List of true label IDs.

        Returns:
            dict[str, float]: Dictionary containing F1 score and accuracy.

        """
        ignore_label_id = -100
        true_predictions = [
            [self.id2label[pred] for (pred, lab) in zip(predictions, labels, strict=False) if lab != ignore_label_id],
        ]
        true_labels = [
            [self.id2label[lab] for (pred, lab) in zip(predictions, labels, strict=False) if lab != ignore_label_id],
        ]

        results = classification_report(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
        acc = accuracy_score(true_labels, true_predictions)
        print(results)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_acc = acc
        return {
            "f1": f1,
            "accuracy": acc,
        }

    def train(self) -> None:
        """Train the classifier using the vectorized training and validation datasets."""
        bs = 128
        num_train_epochs = 5
        train_len = self.ds_len
        steps_per_epoch = math.ceil(train_len / bs)
        max_steps = steps_per_epoch * num_train_epochs

        print(f"Max steps: {max_steps}, Batch size: {bs}, Dataset length: {train_len}")

        train_loader = DataLoader(
            self.vectorized_train,
            batch_size=bs,
            collate_fn=self.collate_fn,
        )
        val_loader = DataLoader(
            self.vectorized_val,
            batch_size=bs * 2,
            collate_fn=self.collate_fn,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=5e-5,
            weight_decay=1e-4,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

        step = 0
        for epoch in range(num_train_epochs):
            self.classifier.train()
            train_loss = 0.0

            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_train_epochs}",
            ):
                step += 1
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = self.classifier(**inputs)
                loss = criterion(outputs["logits"], labels)

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
                            logits = outputs["logits"]
                            preds = torch.argmax(logits, dim=-1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())

                    metrics = self.compute_metrics(all_preds, all_labels)
                    print(f"[Step {step}] Eval metrics: {metrics}")

            avg_train_loss = train_loss / steps_per_epoch
            print(f"Epoch {epoch + 1} completed. Avg train loss: {avg_train_loss:.4f}")

        train_ds = pathlib.Path("./vectorized_train.parquet")
        val_ds = pathlib.Path("./vectorized_val.parquet")
        if train_ds.is_file():
            pathlib.Path.unlink(train_ds)
        if val_ds.is_file():
            pathlib.Path.unlink(val_ds)


class NERValidator(Validator, NEREmbedder):
    """NER_Validator class for training and evaluating a NER classifier using pre-computed embeddings."""

    def __init__(
        self,
        dataset_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cutoff: int = 1000,
    ) -> None:
        """Initialize the NER_Validator with dataset path, model, tokenizer, and cutoff for embedding generation.

        Args:
            dataset_path (str): Path to the dataset.
            model (PreTrainedModel): Pre-trained model for generating embeddings.
            tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the pre-trained model.
            cutoff (int, optional): Maximum number of samples to use for embedding generation. Defaults to 1000.

        """
        NEREmbedder.__init__(self, dataset_path, model, tokenizer, cutoff=cutoff)
        Validator.__init__(
            self,
            self.hidden_size,
            self.ds_len,
            id2label=self.id2label,
            nc=self.num_classes,
        )
        self.vectorized_train, self.vectorized_val = self.get_embeddings()
        self.train()

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate function to prepare batches of embeddings and labels for training and evaluation.

        Args:
            batch (list[dict[str, Any]]): List of samples, where each sample is a dictionary
                containing "embedding" and "labels".

        Returns:
            dict[str, torch.Tensor]: Dictionary containing stacked embeddings and corresponding labels as tensors.

        """
        return {
            "embeddings": torch.stack(
                [(f["embedding"]).clone().detach() for f in batch],
            ),
            "labels": torch.tensor([f["labels"] for f in batch]),
        }


class REValidator(Validator, REEmbedder):
    """RE_Validator class for training and evaluating a RE classifier using pre-computed embeddings."""

    def __init__(
        self,
        dataset_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cutoff: int = 1000,
    ) -> None:
        """Initialize the RE_Validator with dataset path, model, tokenizer, and cutoff for embedding generation.

        Args:
            dataset_path (str): Path to the dataset.
            model (PreTrainedModel): Pre-trained model for generating embeddings.
            tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the pre-trained model.
            cutoff (int, optional): Maximum number of samples to use for embedding generation. Defaults to 1000.

        """
        REEmbedder.__init__(self, dataset_path, model, tokenizer, cutoff=cutoff)
        Validator.__init__(
            self,
            self.hidden_size * 2,
            self.ds_len,
            id2label=self.id2label,
            nc=self.num_classes,
        )
        self.vectorized_train, self.vectorized_val = self.get_embeddings()
        self.train()

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate function to prepare batches of entity pair embeddings and labels for training and evaluation.

        Args:
            batch (list[dict[str, Any]]): List of samples, where each sample is a dictionary
                containing "e1_embedding", "e2_embedding", and "label".

        Returns:
            dict[str, torch.Tensor]: Dictionary containing concatenated entity pair embeddings
            and corresponding labels as tensors.

        """
        e1_embeddings = torch.stack([f["e1_embedding"] for f in batch])
        e2_embeddings = torch.stack([f["e2_embedding"] for f in batch])
        combined_embeddings = torch.cat([e1_embeddings, e2_embeddings], dim=-1)

        labels = torch.tensor([f["label"] for f in batch])

        return {"embeddings": combined_embeddings, "labels": labels}
