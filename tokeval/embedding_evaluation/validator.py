"""Validator module for training and evaluating NER and RE classifiers using pre-computed embeddings."""

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
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
        self.f1_ci_low = 0.0
        self.f1_ci_high = 0.0

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
        seq_preds: list[list] = []
        seq_labels: list[list] = []

        with torch.no_grad():
            for batch in loader:
                embeddings = batch["embeddings"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.classifier(embeddings=embeddings)["logits"]
                preds = torch.argmax(logits, dim=-1)

                if labels.dim() == 1:
                    # Один токен на сэмпл — каждый токен оборачиваем в список
                    for pr, la in zip(preds.cpu().tolist(), labels.cpu().tolist(), strict=False):
                        seq_preds.append([pr])
                        seq_labels.append([la])
                else:
                    # Несколько токенов на сэмпл [batch_size, seq_len]
                    for i in range(labels.size(0)):
                        seq_preds.append(preds[i].cpu().tolist())
                        seq_labels.append(labels[i].cpu().tolist())

        return self._compute_metrics(seq_preds, seq_labels)

    # ------------------ METRICS ------------------

    def _bootstrap_worker(self, args: tuple) -> tuple[float, float]:
        seq_true_labels, seq_true_preds, seed = args
        rng = np.random.default_rng(seed)
        n = len(seq_true_labels)

        indices = rng.integers(0, n, n)
        sample_labels = [seq_true_labels[i] for i in indices]
        sample_preds = [seq_true_preds[i] for i in indices]

        report = classification_report(sample_labels, sample_preds, output_dict=True, zero_division=0)
        f1 = report["micro avg"]["f1-score"]
        acc = accuracy_score(sample_labels, sample_preds)

        return f1, acc

    def _compute_metrics(
        self,
        preds: list[list[int]],
        labels: list[list[int]],
        n_bootstrap: int = 100,
        alpha: float = 0.95,
        n_jobs: int = 4,
    ) -> dict[str, float]:
        ignore_label = -100

        seq_true_labels: list[list[str]] = []
        seq_true_preds: list[list[str]] = []

        for pred_seq, label_seq in zip(preds, labels, strict=False):
            filtered_labels, filtered_preds = [], []
            for pr, la in zip(pred_seq, label_seq, strict=False):
                if la == ignore_label:
                    continue
                filtered_labels.append(self.id2label[la])
                filtered_preds.append(self.id2label[pr])
            if filtered_labels:
                seq_true_labels.append(filtered_labels)
                seq_true_preds.append(filtered_preds)

        # --- точечная оценка ---
        report = classification_report(seq_true_labels, seq_true_preds, output_dict=True, zero_division=0)
        f1 = report["micro avg"]["f1-score"]
        acc = accuracy_score(seq_true_labels, seq_true_preds)

        # --- параллельный bootstrap ---
        seeds = np.random.SeedSequence().spawn(n_bootstrap)
        tasks = [(seq_true_labels, seq_true_preds, int(s.generate_state(1)[0])) for s in seeds]

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
            results = list(pool.map(self._bootstrap_worker, tasks))

        f1_scores, acc_scores = zip(*results, strict=False)

        # --- доверительные интервалы ---
        lower_q = (1 - alpha) / 2
        upper_q = 1 - lower_q

        f1_ci = np.quantile(f1_scores, [lower_q, upper_q])
        acc_ci = np.quantile(acc_scores, [lower_q, upper_q])

        print(f"Sentences evaluated : {len(seq_true_labels)}")
        print(f"Point F1            : {f1:.4f}")
        print(f"Bootstrap mean F1   : {np.mean(f1_scores):.4f}  (Δ={abs(f1 - np.mean(f1_scores)):.4f})")
        print(f"95% CI F1           : [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")

        return {
            "f1": f1,
            "f1_ci_low": float(f1_ci[0]),
            "f1_ci_high": float(f1_ci[1]),
            "accuracy": acc,
            "accuracy_ci_low": float(acc_ci[0]),
            "accuracy_ci_high": float(acc_ci[1]),
        }

    # ------------------ BEST METRIC TRACKING ------------------

    def _update_best(self, metrics: dict[str, float]) -> None:
        if metrics["f1"] > self.best_f1:
            self.best_f1 = metrics["f1"]
            self.f1_ci_low = metrics["f1_ci_low"]
            self.f1_ci_high = metrics["f1_ci_high"]
            self.best_acc = metrics["accuracy"]

    def get_results(self) -> tuple[float, float, float, float]:
        """Return the best F1 score and accuracy achieved during training.

        Returns:
            tuple of floats with F1-score, its lower CI bound, upper CI bound, and accuracy

        """
        return self.best_f1, self.f1_ci_low, self.f1_ci_high, self.best_acc


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

    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.classifier.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in loader:
                embeddings = batch["embeddings"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.classifier(embeddings=embeddings)["logits"]
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        return self._compute_metrics(all_preds, all_labels)

    def _re_bootstrap_worker(self, args: tuple) -> tuple[float, float]:
        """Одна bootstrap-итерация для RE. Запускается в отдельном процессе."""
        true_labels, true_preds, seed = args
        rng = np.random.default_rng(seed)
        n = len(true_labels)

        indices = rng.integers(0, n, n)
        sample_labels = true_labels[indices]
        sample_preds = true_preds[indices]

        report = sklearn_classification_report(
            sample_labels,
            sample_preds,
            output_dict=True,
            zero_division=0,
        )
        return report["weighted avg"]["f1-score"], sklearn_accuracy_score(sample_labels, sample_preds)

    def _compute_metrics(
        self,
        preds: list[int],
        labels: list[int],
        n_bootstrap: int = 100,
        alpha: float = 0.95,
        n_jobs: int = 8,
    ) -> dict[str, float]:
        ignore_label = -100

        true_preds: list[str] = []
        true_labels: list[str] = []

        for pr, la in zip(preds, labels, strict=False):
            if la == ignore_label:
                continue
            true_preds.append(self.id2label[pr])
            true_labels.append(self.id2label[la])

        true_preds_arr = np.array(true_preds)
        true_labels_arr = np.array(true_labels)

        # --- точечная оценка ---
        report = sklearn_classification_report(
            true_labels_arr,
            true_preds_arr,
            output_dict=True,
            zero_division=0,
        )
        f1 = report["weighted avg"]["f1-score"]
        acc = sklearn_accuracy_score(true_labels_arr, true_preds_arr)

        # --- параллельный bootstrap ---
        seeds = np.random.SeedSequence().spawn(n_bootstrap)
        tasks = [(true_labels_arr, true_preds_arr, int(s.generate_state(1)[0])) for s in seeds]

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
            results = list(pool.map(self._re_bootstrap_worker, tasks))

        f1_scores, acc_scores = zip(*results, strict=False)

        # --- доверительные интервалы ---
        lower_q = (1 - alpha) / 2
        upper_q = 1 - lower_q

        f1_ci = np.quantile(f1_scores, [lower_q, upper_q])
        acc_ci = np.quantile(acc_scores, [lower_q, upper_q])

        print(f"Samples evaluated   : {len(true_labels)}")
        print(f"Point F1            : {f1:.4f}")
        print(f"Bootstrap mean F1   : {np.mean(f1_scores):.4f}  (Δ={abs(f1 - np.mean(f1_scores)):.4f})")
        print(f"95% CI F1           : [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")

        return {
            "f1": f1,
            "f1_ci_low": float(f1_ci[0]),
            "f1_ci_high": float(f1_ci[1]),
            "accuracy": acc,
            "accuracy_ci_low": float(acc_ci[0]),
            "accuracy_ci_high": float(acc_ci[1]),
        }

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
