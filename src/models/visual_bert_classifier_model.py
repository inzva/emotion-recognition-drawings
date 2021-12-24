from typing import Any
import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, VisualBertModel
from typing import Optional
from src.datamodules.datasets.dataset_output import DatasetOutput
from src.models.modules.visual_bert_classifier import VisualBertClassifier
from src.utils.metric.multi_label_classification_metric import MultiLabelClassificationMetric


class VisualBertClassifierLitModel(LightningModule):
    def __init__(self,
                 num_classes: int,
                 num_train_steps: float,
                 dataset_output: int = 3,
                 dropout_rate: float = 0.2,
                 lr: float = 0.001,
                 weight_decay: float = 0.0005,
                 use_scheduler: bool = True,
                 initial_visual_embedding_dim: int = 96,
                 scheduler_num_warmup_steps: int = 0,
                 # This will only be used to get inner Visual BERT model,
                 # the final linear layer will be instantiated from scratch
                 pretrained_lit_model_for_body_checkpoint: Optional[str] = None,
                 visual_bert_model_name: str = "uclanlp/visualbert-vqa-coco-pre"
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_output = DatasetOutput(dataset_output)
        if pretrained_lit_model_for_body_checkpoint is not None:
            pretrained_lit_model = VisualBertClassifierLitModel.load_from_checkpoint(
                pretrained_lit_model_for_body_checkpoint)
            visual_bert_model = pretrained_lit_model.model.visual_bert
        elif visual_bert_model_name == "uclanlp/visualbert-vqa-coco-pre":
            visual_bert_model = VisualBertModel.from_pretrained(visual_bert_model_name)
        else:
            raise Exception(
                "Unknown visual_bert_model_name for VisualBertClassifierLitModel"
            )
        self.model = VisualBertClassifier(visual_bert_model=visual_bert_model,
                                          num_classes=num_classes,
                                          initial_visual_embedding_dim=initial_visual_embedding_dim,
                                          final_dropout_rate=dropout_rate)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.metric = MultiLabelClassificationMetric.get_implementation(MultiLabelClassificationMetric.RocCurve_Scikit)

    def forward(self,
                text_input_ids,
                text_token_type_ids,
                text_attention_mask,
                visual_embeds,
                visual_token_type_ids,
                visual_attention_mask):
        return self.model(text_input_ids,
                          text_token_type_ids,
                          text_attention_mask,
                          visual_embeds,
                          visual_token_type_ids,
                          visual_attention_mask)

    def step(self, batch: Any):
        if self.dataset_output is DatasetOutput.EmoRecComVisualBertOutput:
            transformer_inputs = batch[3]
            text_input_ids, text_attention_mask, text_token_type_ids = transformer_inputs["ids"], transformer_inputs[
                "mask"], \
                                                                       transformer_inputs["token_type_ids"]
            targets, polarities = batch[2]
            visual_embeds, visual_token_type_ids, visual_attention_mask = batch[4]
        else:
            raise Exception(
                "Invalid dataset output for BertClassifierLitModel"
            )
        scores = self.forward(text_input_ids,
                              text_token_type_ids,
                              text_attention_mask,
                              visual_embeds,
                              visual_token_type_ids,
                              visual_attention_mask)
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

    # TODO: @all read this https://www.fast.ai/2018/07/02/adam-weight-decay/ to understand AdamW
    #   i have not read yet @gsoykan
    # source: https://github.dev/abhishekkrthakur/tez/blob/main/examples/text_classification/binary.py
    # configuring optimizers with Lightning: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
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
