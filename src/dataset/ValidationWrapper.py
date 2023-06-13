from random import randint
from typing import Iterator, Optional

from torch.utils.data import Dataset, IterableDataset


class ValidationWrapper(Dataset):
    """Wraps a dataset so that PyTorch Lightning's validation step can be turned into a
    visualization step.
    """

    dataset: Dataset
    dataset_iterator: Optional[Iterator]
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length
        self.dataset_iterator = None
        if isinstance(dataset, IterableDataset):
            self.dataset_iterator = iter(dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if self.dataset_iterator is not None:
            return next(self.dataset_iterator)
        return self.dataset[randint(0, len(self.dataset) - 1)]
