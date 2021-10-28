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
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def replace_with_glove_embeddings(self,
                                      glove_weights,
                                      glove_embedding_dim: int = 50):
        self.embeddings = nn.Embedding(self.vocab_size, glove_embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        self.embeddings.weight.requires_grad = False  ## freeze embeddings
        # reinstantiating LSTM with glove dim embedding size
        self.lstm = nn.LSTM(glove_embedding_dim, self.hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])
