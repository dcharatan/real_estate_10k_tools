import os
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.dataset.DataModule import DataModule
    from src.misc.wandb_tools import download_latest_checkpoint
    from src.model.ModelWrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg: DictConfig):
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=f"{cfg.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            log_model="all",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code(".")
    else:
        logger = None

    # Set up checkpointing.
    if "checkpointing" in cfg.train:
        callbacks.append(
            ModelCheckpoint(
                output_dir / "checkpoints",
                **cfg.train.checkpointing,
            )
        )

    # Prepare the checkpoint for loading.
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        checkpoint_path = None
    elif str(checkpoint).startswith("wandb://"):
        run_id = checkpoint[len("wandb://") :]
        project = cfg.wandb.project
        checkpoint_path = download_latest_checkpoint(
            f"{project}/{run_id}",
            Path("checkpoints"),
        )
    else:
        checkpoint_path = Path(checkpoint)

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.val.interval,
    )
    trainer.fit(
        ModelWrapper(cfg),
        datamodule=DataModule(cfg),
        ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":
    train()
