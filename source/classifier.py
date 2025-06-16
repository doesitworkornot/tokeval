import torch.nn as nn



class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=32):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.silu = nn.SiLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels=None):
        x = self.batch_norm1(embeddings)
        x = self.linear1(x)
        x = self.batch_norm2(x)
        x = self.silu(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}