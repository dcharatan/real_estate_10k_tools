import torch
from omegaconf import DictConfig
from torch import nn


class ModelPlaceholder(nn.Module):
    cfg_model: DictConfig

    def __init__(self, cfg_model: DictConfig) -> None:
        super().__init__()
        self.cfg_model = cfg_model
        self.placeholder = nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self):
        return self.placeholder
