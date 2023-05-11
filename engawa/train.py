from enum import Enum
from typing import Annotated

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast

from engawa.data_loader import get_dataloader
from engawa.model import EngawaModel


class ModelType(str, Enum):
    base = "base"
    large = "large"


def train_model(
    tokenizer_file: Annotated[str, typer.Option(help="Path to a tokenizer file.")],
    train_file: Annotated[
        str, typer.Option(help="Line separated text file for training.")
    ],
    val_file: Annotated[
        str, typer.Option(help="Line separated text file for validation.")
    ],
    default_root_dir: Annotated[
        str, typer.Option(help="Path to save generated files.")
    ],
    ckpt_path: Annotated[
        str, typer.Option(help="If given, start from that checkpoint.")
    ] = None,
    wandb_proj_name: Annotated[
        str, typer.Option(help="If given, log to that wandb project.")
    ] = None,
    val_check_interval: float = 0.25,
    max_steps: Annotated[
        int, typer.Option(help="Set -1 for infinite training.")
    ] = 500000,
    model_type: Annotated[
        ModelType, typer.Option(help="BART type, `base` or `large`.")
    ] = ModelType.large,
    seed: int = 10,
    bs: int = 32,
    max_length: int = 1024,
    lr: float = 0.0004,
    weight_decay: float = 0.01,
    num_warmup_steps: int = 10000,
    mask_ratio: float = 0.3,
    poisson_lambda: float = 3.5,
):
    tokenizer = BartTokenizerFast(tokenizer_file=tokenizer_file)

    pl.seed_everything(seed)

    train_dl = get_dataloader(
        train_file,
        "train",
        bs,
        tokenizer,
        mask_ratio,
        poisson_lambda,
        max_length=max_length,
    )
    val_dl = get_dataloader(
        val_file,
        "validation",
        bs,
        tokenizer,
        mask_ratio,
        poisson_lambda,
        max_length=max_length,
    )

    if wandb_proj_name is not None:
        logger = WandbLogger(project=wandb_proj_name, save_dir=default_root_dir)
    else:
        logger = CSVLogger(default_root_dir)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    num_gpus = torch.cuda.device_count()
    devices = None if num_gpus == 0 else num_gpus
    accelerator = "cpu" if num_gpus == 0 else "gpu"

    trainer = pl.Trainer(
        max_epochs=-1,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        deterministic=True,
        default_root_dir=default_root_dir,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    )

    model = EngawaModel(
        tokenizer,
        lr,
        weight_decay,
        num_warmup_steps,
        max_steps if max_steps > -1 else 500000,
        model_type=model_type,
    )

    if ckpt_path is not None:
        print(f"Resume training from {ckpt_path}...")
    trainer.fit(model, train_dl, val_dl, ckpt_path=ckpt_path)
