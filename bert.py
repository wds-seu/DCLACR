from typing import Optional, Union, List

import torch.optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn
from torch.nn import functional as F
from torchmetrics import F1Score
from transformers import BertModel


class BERT(LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.save_hyperparameters(hp)

        self.train_f1 = F1Score(hp.n_labels, 0.0, average='micro')
        self.val_f1 = F1Score(hp.n_labels, 0.0, average='macro')

        self.bert: BertModel = BertModel.from_pretrained(
            "./biobert"
        )

        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            5
        )

    def forward(self, batch):
        input_ids = batch.text.input_ids.squeeze(dim=1)
        token_type_ids = batch.text.token_type_ids.squeeze(dim=1)
        attention_mask = batch.text.attention_mask.squeeze(dim=1)
        model_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = model_output.last_hidden_state
        mask = ((input_ids == 1) + (input_ids == 2) + (input_ids == 3)).unsqueeze(dim=-1)
        pooled_output = (mask * last_hidden_state).sum(dim=1)
        y_scores = self.classifier(pooled_output)
        if self.trainer.testing:
            loss = None
        else:
            #batch.labels.to(y_scores)
            loss = F.cross_entropy(y_scores, batch.label )
        return loss, y_scores

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer], [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)
        ]

    #"labels": batch.labels,
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, preds = self(batch)
        return {
            "loss": loss,
            "preds": preds,
            "label": batch.label,
        }

    def training_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        #self.val_f1.update(output["preds"], output["labels"].long())
        self.train_f1.update(output["preds"], output["label"].long())
        self.log("f1", self.train_f1.compute(), on_epoch=True, on_step=True, prog_bar=True)
        return output

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.train_f1.reset()
    #"labels": batch.labels,
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss, preds = self(batch)
        return {
            "loss": loss,
            "preds": preds,
            "label": batch.label,
        }

    def validation_step_end(self, output: STEP_OUTPUT) -> Optional[STEP_OUTPUT]:
        #self.val_f1.update(output["preds"], output["labels"].long())
        self.val_f1.update(output["preds"], output["label"].long())
        self.log("val_f1", self.val_f1.compute(), on_epoch=True, on_step=False, prog_bar=True)
        return output

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.val_f1.reset()

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        _, preds = self(batch)
        return {
            "preds": preds
        }

    def test_step_end(self, output) -> Optional[STEP_OUTPUT]:
        return output
