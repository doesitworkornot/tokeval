"""Validator module for training and evaluating NER and RE classifiers using pre-computed embeddings."""

from collections.abc import Callable

import torch
from seqeval.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import classification_report as sklearn_classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tokeval.embedding_evaluation.classifier import Classifier
from tokeval.embedding_evaluation.embeddings import NEREmbedder, REEmbedder


class Validator:
    """Base Validator class for training and evaluating token classification models."""

    def __init__(
        self,
        hidden_size: int,
        id2label: dict[int, str],
        train_ds: Dataset,
        val_ds: Dataset,
        collate_fn: Callable,
    ) -> None:
        """Initialize Validator class for training and evaluating token classification models.

        Args:
            hidden_size: The dimension of the input embeddings.
            id2label: A dictionary mapping label IDs to label names.
            train_ds: The training dataset containing pre-computed embeddings and labels.
            val_ds: The validation dataset containing pre-computed embeddings and labels.
            collate_fn: A function to collate batches of data from the datasets.

        """
        epochs: int = 5
        batch_size: int = 128
        lr: float = 5e-5

        self.id2label = id2label
        self.label_list = [id2label[i] for i in sorted(id2label)]

        num_classes = len(self.label_list)
        self.classifier = Classifier(hidden_size, num_classes)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.collate_fn = collate_fn

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.best_f1 = 0.0
        self.best_acc = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)

    # ------------------ TRAIN LOOP ------------------

    def train(self) -> None:
        """Train the classifier.

        Train on the training dataset and evaluate on the validation dataset after each epoch,
        tracking the best F1 score and accuracy.
        """
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size * 3,
            collate_fn=self.collate_fn,
        )

        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            metrics = self._evaluate(val_loader)

            print(
                f"\nEpoch {epoch + 1}/{self.epochs} "
                f"| train_loss={train_loss:.4f} "
                f"| val_f1={metrics['f1']:.4f} "
                f"| val_acc={metrics['accuracy']:.4f}",
            )

            self._update_best(metrics)

    # ------------------ TRAIN STEP ------------------

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.AdamW, criterion: nn.CrossEntropyLoss) -> float:
        self.classifier.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training"):
            embeddings = batch["embeddings"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.classifier(embeddings=embeddings)
            logits = outputs["logits"]

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss

    # ------------------ EVALUATION ------------------

    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.classifier.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                embeddings = batch["embeddings"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.classifier(embeddings=embeddings)["logits"]
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        return self._compute_metrics(all_preds, all_labels)

    # ------------------ METRICS ------------------

    def _compute_metrics(self, preds: list, labels: list) -> dict[str, float]:
        ignore_label = -100

        true_preds = []
        true_labels = []

        for pr, la in zip(preds, labels, strict=False):
            if la == ignore_label:
                continue
            true_preds.append(self.id2label[pr])
            true_labels.append(self.id2label[la])

        # классификационный отчет
        report = classification_report(
            [true_labels],
            [true_preds],
            output_dict=True,
            zero_division=0,
        )

        f1 = report["micro avg"]["f1-score"]
        acc = accuracy_score(true_labels, true_preds)

        return {"f1": f1, "accuracy": acc}

    # ------------------ BEST METRIC TRACKING ------------------

    def _update_best(self, metrics: dict[str, float]) -> None:
        if metrics["f1"] > self.best_f1:
            self.best_f1 = metrics["f1"]
            self.best_acc = metrics["accuracy"]

    def get_results(self) -> tuple[float, float]:
        """Return the best F1 score and accuracy achieved during training.

        Returns:
            A tuple containing the best F1 score and best accuracy.

        """
        return self.best_f1, self.best_acc


class NERValidator(Validator):
    """NERValidator uses the Classifier for token classification tasks with NER embeddings."""

    def __init__(self, embedder: NEREmbedder) -> None:
        """NERValidator uses the Classifier for token classification tasks with NER embeddings.

        Args:
            embedder: An instance of NEREmbedder that provides the training and
                validation datasets with pre-computed NER embeddings.

        """
        train_ds, val_ds = embedder.get_embeddings()

        super().__init__(
            hidden_size=embedder.hidden_size,
            id2label=embedder.id2label,
            train_ds=train_ds,
            val_ds=val_ds,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate function for RE embeddings, concatenating entity embeddings.

        Args:
            batch: A list of samples, where each sample is a dictionary containing 'embedding' and 'labels'.

        Returns:
            A dictionary with 'embeddings' as a tensor of shape (batch_size, seq_length, hidden_size)
                and 'labels' as a tensor of shape (batch_size, seq_length).

        """
        return {
            "embeddings": torch.stack([f["embedding"] for f in batch]),
            "labels": torch.tensor([f["labels"] for f in batch]),
        }


class REValidator(Validator):
    """Class for Validation of token embedders.

    Uses the same Classifier as NERValidator but with a different collate function to handle RE embeddings.
    """

    def __init__(self, embedder: REEmbedder) -> None:
        """Initialize the REValidator with the provided REEmbedder.

        Args:
            embedder: An instance of REEmbedder that provides the training
                and validation datasets with pre-computed RE embeddings.

        """
        train_ds, val_ds = embedder.get_embeddings()

        super().__init__(
            hidden_size=embedder.hidden_size * 2,
            id2label=embedder.id2label,
            train_ds=train_ds,
            val_ds=val_ds,
            collate_fn=self.collate_fn,
        )

    def _compute_metrics(self, preds: list, labels: list) -> dict[str, float]:
        ignore_label = -100

        # одномерные списки классов
        true_preds = []
        true_labels = []

        for pr, la in zip(preds, labels, strict=False):
            if la == ignore_label:
                continue
            true_preds.append(self.id2label[pr])
            true_labels.append(self.id2label[la])

        # sklearn подходит для RE (одиночные классы)
        report = sklearn_classification_report(true_labels, true_preds, output_dict=True, zero_division=0)
        f1 = report["weighted avg"]["f1-score"]
        acc = sklearn_accuracy_score(true_labels, true_preds)

        return {"f1": f1, "accuracy": acc}

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate function for RE embeddings, concatenating entity embeddings.

        Args:
            batch: A list of samples, where each sample is a dictionary
                containing 'e1_embedding', 'e2_embedding', and 'label'.

        Returns:
            A dictionary with 'embeddings' as a tensor of shape (batch_size, hidden_size * 2)
                and 'labels' as a tensor of shape (batch_size).

        """
        e1 = torch.stack([f["e1_embedding"] for f in batch])
        e2 = torch.stack([f["e2_embedding"] for f in batch])
        labels = torch.tensor([f["label"] for f in batch])

        return {
            "embeddings": torch.cat([e1, e2], dim=-1),
            "labels": labels,
        }
