from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning as L
import torch
from torch import nn

from vision_transformers_cifar.models import VisionTransformer
from vision_transformers_cifar.modules.types import OptimizerType

if TYPE_CHECKING:
    from vision_transformers_cifar.config import (
        OptimizerConfig,
        SchedulerConfig,
        ViTModelConfig,
    )


class VisionTransformerLightningModule(L.LightningModule):
    def __init__(
        self,
        model_config: ViTModelConfig | dict[str, Any],
        optimizer_config: OptimizerConfig | dict[str, Any],
        scheduler_config: SchedulerConfig | dict[str, Any],
    ) -> None:
        from vision_transformers_cifar.config import (
            OptimizerConfig,
            SchedulerConfig,
            ViTModelConfig,
        )

        super().__init__()

        self.model_config = ViTModelConfig.model_validate(model_config)
        self.optimizer_config = OptimizerConfig.model_validate(optimizer_config)
        self.scheduler_config = SchedulerConfig.model_validate(scheduler_config)
        self.save_hyperparameters(
            {
                "model_config": self.model_config.model_dump(mode="json"),
                "optimizer_config": self.optimizer_config.model_dump(mode="json"),
                "scheduler_config": self.scheduler_config.model_dump(mode="json"),
            }
        )
        self.model = VisionTransformer(**self.model_config.model_dump())

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer_cls = {OptimizerType.ADAMW: torch.optim.AdamW, OptimizerType.ADAM: torch.optim.Adam}[
            self.optimizer_config.name
        ]
        optimizer = optimizer_cls(self.parameters(), **self.optimizer_config.model_dump(exclude={"name"}))
        if not self.scheduler_config.enabled:
            return optimizer

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}}

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
