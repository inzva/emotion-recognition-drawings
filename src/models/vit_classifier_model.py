from typing import Any
import torch
import transformers
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Optional
from src.datamodules.datasets.dataset_output import DatasetOutput
from src.models.modules.bert_classifier import BertClassifier
from src.models.modules.vit_classifier import ViTClassifier
from src.utils.metric.micro_auc import compute_micro_auc
from src.utils.metric.multi_label_classification_metric import MultiLabelClassificationMetric


class ViTClassifierLitModel(LightningModule):
    def __init__(self,
                 num_classes: int,
                 num_train_steps: float,
                 dataset_output: int = 4,
                 lr: float = 0.001,
                 weight_decay: float = 0.0005,
                 use_scheduler: bool = True,
                 scheduler_num_warmup_steps: int = 0,
                 feature_extractor_alias: str = "google/vit-base-patch16-224",
                 model_alias: str = "google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_output = DatasetOutput(dataset_output)
        self.model = ViTClassifier(feature_extractor_alias, model_alias, num_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.metric = MultiLabelClassificationMetric.get_implementation(MultiLabelClassificationMetric.RocCurve_Scikit)

    def forward(self, x):
        return self.model(x)

    def step(self, batch: Any):
        if self.dataset_output is DatasetOutput.EmoRecComVisionOutput:
            image_tensor = batch[0]
            targets, polarities = batch[2]
        else:
            raise Exception(
                "Invalid dataset output for BertClassifierLitModel"
            )
        scores = self.forward(image_tensor)
        loss = self.criterion(scores, targets.float())
        preds = F.sigmoid(scores)
        return loss, preds, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        roc_auc_score = self.metric(preds, targets)
        self.log("train/bce_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/roc_auc_score", roc_auc_score, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        roc_auc_score = self.metric(preds, targets)
        self.log("val/bce_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/roc_auc_score", roc_auc_score, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        roc_auc_score = self.metric(preds, targets)
        self.log("test/bce_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/roc_auc_score", roc_auc_score, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optim_params = {}
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                # 0.001 in original case
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # lr: 3e-5 in original case
        optimizer = AdamW(optimizer_parameters,
                          lr=self.hparams.lr)
        optim_params["optimizer"] = optimizer
        # n_train_steps = int(len(train_dataset) / config.batch_size * num_epoch)
        if self.hparams.use_scheduler:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.hparams.scheduler_num_warmup_steps,
                                                        num_training_steps=int(self.hparams.num_train_steps))
            optim_params["lr_scheduler"] = scheduler
        return optim_params
