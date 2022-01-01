from typing import Any
import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, VisualBertModel
from typing import Optional
from src.datamodules.datasets.dataset_output import DatasetOutput
from src.utils.metric.multi_label_classification_metric import MultiLabelClassificationMetric
import clip

from torch import nn as nn
from src.utils.text.text_utils import merge_comic_texts

class ClipClassifierModel(LightningModule):
    def __init__(self,
                 num_classes: int,
                 dropout_rate: float = 0.2,
                 num_train_steps=100,
                 lr: float = 0.001,
                 weight_decay: float = 0.0005,
                 use_scheduler: bool = True,
                 scheduler_num_warmup_steps: int = 0,
                 clip_model: str ="R50",
                 
                 ):
        super().__init__()

        assert clip_model in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
        self.save_hyperparameters()



        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.metric = MultiLabelClassificationMetric.get_implementation(MultiLabelClassificationMetric.RocCurve_Scikit)

        # I am not sure is there any other way to reach current device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=device)

        self.classifier = torch.nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024,num_classes))


    def forward(self, batch):

        img, sizes, (labels, targets), merged_text = batch
        
        text_encoding = self.clip_model.encode_text(clip.tokenize(merged_text, truncate=True))
        image_encoding = self.clip_model.encode_image(img)
        feature_vec = torch.cat( [text_encoding, image_encoding] , dim=1)

        scores = self.classifier(feature_vec)
        return scores

    def step(self, batch):
        img, sizes, (labels, targets), merged_text = batch

        scores = self.forward(batch)
        loss = self.criterion(scores, targets.float())
        preds = F.sigmoid(scores)

        return loss, preds, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        micro_auc = self.metric(preds, targets)
        self.log("train/bce_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/roc_auc_score", micro_auc, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        micro_auc = self.metric(preds, targets)
        self.log("val/bce_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/roc_auc_score", micro_auc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        micro_auc = self.metric(preds, targets)
        self.log("test/bce_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/roc_auc_score", micro_auc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}


    def configure_optimizers(self):
        optim_params = {}


        optimizer = AdamW(self.parameters(),
                        lr=self.hparams.lr)
        optim_params["optimizer"] = optimizer
        # n_train_steps = int(len(train_dataset) / config.batch_size * num_epoch)
        if self.hparams.use_scheduler:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.hparams.scheduler_num_warmup_steps,
                                                        num_training_steps=int(self.hparams.num_train_steps))
            optim_params["lr_scheduler"] = scheduler
        return optim_params




            
           




