from typing import Any

import lightning as L
import torch
from torch import nn

from vision_transformers_cifar.models import VisionTransformer


class VisionTransformerLightningModule(L.LightningModule):
    def __init__(self, model_kwargs: dict[str, Any], lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [lr_scheduler]

    def _common_step(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_fn(preds, labels)

        accuracy = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{mode}_acc", accuracy, prog_bar=True, on_epoch=True)
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, mode="test")
