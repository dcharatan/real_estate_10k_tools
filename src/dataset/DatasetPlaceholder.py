import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .types import Stage


class DatasetPlaceholder(Dataset):
    def __init__(self, cfg_dataset: DictConfig, stage: Stage) -> None:
        super().__init__()
        self.cfg_dataset = cfg_dataset

    def __getitem__(self, index: int):
        return torch.zeros((64, 64), dtype=torch.float32)

    def __len__(self) -> int:
        return 65536
