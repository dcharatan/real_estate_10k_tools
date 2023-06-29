import os
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        os.system(f"rm -r {LOG_PATH}")

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(self, key: str, images: List[Any], step: Optional[int] = None):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        LOG_PATH.mkdir(exist_ok=True, parents=True)
        key = key.replace("/", "_")
        for index, image in enumerate(images):
            Image.fromarray(image).save(LOG_PATH / f"{key}_{index:0>2}_{step:0>6}.png")
