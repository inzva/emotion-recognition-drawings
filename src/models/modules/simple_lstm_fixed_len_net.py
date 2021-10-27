import torch
from torch import nn


class SimpleLSTMFixedLenNet(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 128,
                 num_classes: int = 8,
                 dropout_rate: float = 0.2):
        super().__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])
