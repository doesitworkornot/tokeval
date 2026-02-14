"""Classifier module for token classification tasks."""

from typing import Any

import torch
from torch import nn


class Classifier(nn.Module):
    """A simple classifier for token classification tasks."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        """Initialize the classifier.

        Args:
            input_dim: The dimension of the input embeddings.
            num_classes: The number of classes for classification.
            hidden_dim: The dimension of the hidden layer (default: 32).

        """
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.silu = nn.SiLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass of the classifier.

        Args:
            embeddings: The input embeddings of shape (batch_size, seq_length, input_dim).
            labels: The true labels of shape (batch_size, seq_length) (optional).

        Returns:
            A dictionary containing the loss (if labels are provided) and the logits.

        """
        x = self.batch_norm1(embeddings)
        x = self.linear1(x)
        x = self.batch_norm2(x)
        x = self.silu(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}
