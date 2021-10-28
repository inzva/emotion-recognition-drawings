from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.auroc import AUROC
from torch.nn import functional as F
from src.models.modules.simple_lstm_fixed_len_net import SimpleLSTMFixedLenNet

"""
3.4 Evaluation metric
The submissions have been evaluated based on the Area Under 
the ReceiverOperating Characteristic Curve (ROC-AUC) score.
The ROC curve, is a graphical plot which illustrates the
performance of a binary classifier system as 
its discrimination threshold is varied. 
The Area Under the ROC Curve (AUC) summarizes the curve
information in one number. 
The ROC-AUC was calculated between the list of predicted
emotions for each image given by the participants and
its corresponding target in the ground
truth (as described in Section 2.2).
This is a multi-class classification where the chosen averaging strategy was 
one-vs-one algorithm that computes the pairwise ROC scores and then the average
of the 8 AUCs, for each image [5]. 
In other words, the score is the average of the individual 
AUCs of each predicted emotion. To compute this score, 
we use the Scikit-learn implementation11.
"""


class SimpleLSTMFixedLenLitModel(LightningModule):
    def __init__(
            self,
            vocab_size: int = 6900,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            num_classes: int = 8,
            dropout_rate: float = 0.2,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleLSTMFixedLenNet(vocab_size,
                                           embedding_dim,
                                           hidden_dim,
                                           num_classes,
                                           dropout_rate)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_accuracy = AUROC(pos_label=1, )
        self.val_accuracy = AUROC(pos_label=1)
        self.test_accuracy = AUROC(pos_label=1)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        scores = self.forward(x)
        loss = self.criterion(scores, y)
        preds = F.sigmoid(scores)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.train_accuracy(preds, targets)
        self.log("train/bce_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auroc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.val_accuracy(preds, targets)
        self.log("val/bce_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auroc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.test_accuracy(preds, targets)
        self.log("test/bce_loss", loss, on_step=False, on_epoch=True)
        self.log("test/auroc", acc, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )