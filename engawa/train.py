import click
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast

from engawa.data_loader import get_dataloader
from engawa.model import EngawaModel


@click.command()
@click.option(
    "--tokenizer-file", type=str, required=True, help="Path to a tokenizer file."
)
@click.option(
    "--train-file",
    type=str,
    required=True,
    help="Line separated text file for training.",
)
@click.option(
    "--val-file",
    type=str,
    required=True,
    help="Line separated text file for validation.",
)
@click.option(
    "--default-root-dir", type=str, required=True, help="Path to save generated files."
)
@click.option(
    "--ckpt-path",
    type=str,
    required=False,
    default=None,
    help="If given, start from that checkpoint.",
)
@click.option(
    "--wandb-proj-name",
    type=str,
    required=False,
    default=None,
    help="If given, log to that wandb project.",
)
@click.option(
    "--val-check-interval",
    type=float,
    required=False,
    default=0.25,
)
@click.option(
    "--max-steps",
    type=int,
    required=False,
    default=500000,
    help="Set -1 for infinite training.",
)
@click.option(
    "--model-type",
    type=click.Choice(
        ["base", "large"],
        case_sensitive=False,
    ),
    required=False,
    default="large",
    help="BART type, `base` or `large`.",
)
@click.option("--seed", type=int, required=False, default=10)
@click.option("--bs", type=int, required=False, default=32)
@click.option("--max-length", type=int, required=False, default=1024)
@click.option("--lr", type=float, required=False, default=0.0004)
@click.option("--weight-decay", type=float, required=False, default=0.01)
@click.option("--num-warmup-steps", type=int, required=False, default=10000)
@click.option("--mask-ratio", type=float, required=False, default=0.3)
@click.option("--poisson_lambda", type=float, required=False, default=3.5)
def train_model(
    tokenizer_file: str,
    train_file: str,
    val_file: str,
    default_root_dir: str,
    ckpt_path: str,
    wandb_proj_name: str,
    val_check_interval: float,
    max_steps: int,
    model_type: str,
    seed: int,
    bs: int,
    max_length: int,
    lr: float,
    weight_decay: float,
    num_warmup_steps: int,
    mask_ratio: float,
    poisson_lambda: float,
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
