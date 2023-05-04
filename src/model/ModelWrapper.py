from typing import Dict, Optional

import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn, optim

from .ModelPlaceholder import ModelPlaceholder

MODELS: Dict[str, nn.Module] = {
    "placeholder": ModelPlaceholder,
}


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: nn.Module
    cfg: DictConfig

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Set up the model.
        self.model = MODELS[cfg.model.name](cfg.model)

    def training_step(self, batch, batch_idx):
        # Compute loss.
        x = self.model()
        loss = x**2

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.cfg.wandb.mode == "disabled":
            return

        self.logger.log_image(
            "placeholder",
            [np.zeros((64, 64, 3), dtype=np.float32)],
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.train.optim.lr)
        return {
            "optimizer": optimizer,
        }
