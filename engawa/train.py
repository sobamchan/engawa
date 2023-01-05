from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast

from engawa.data_loader import get_dataloader
from engawa.model import EngawaModel

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--tokenizer-file", type=str, required=True, help="Your `tokenizer.json`"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Line separated text file for training.",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        required=True,
        help="Line separated text file for validation.",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        required=True,
        help="Path to save generated files.",
    )
    parser.add_argument("--ckpt-path", type=str, required=False, default=None)
    parser.add_argument("--wandb-proj-name", type=str, required=False, default=None)
    parser.add_argument(
        "--val-check-interval", type=float, required=False, default=0.25
    )

    # parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument(
        "--max-steps", type=int, default=500000, help="Set -1 for infinite training."
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["base", "large"],
        default="large",
        help="BART size, `base` or `large`.",
    )

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-warmup-steps", type=int, default=10000)

    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--poisson_lambda", type=float, default=3.5)

    args = parser.parse_args()

    tokenizer = BartTokenizerFast(tokenizer_file=args.tokenizer_file)

    pl.seed_everything(args.seed)

    train_dl = get_dataloader(
        args.train_file,
        "train",
        args.bs,
        tokenizer,
        args.mask_ratio,
        args.poisson_lambda,
        max_length=args.max_length,
    )
    val_dl = get_dataloader(
        args.val_file,
        "validation",
        args.bs,
        tokenizer,
        args.mask_ratio,
        args.poisson_lambda,
        max_length=args.max_length,
    )

    if args.wandb_proj_name is not None:
        logger = WandbLogger(
            project=args.wandb_proj_name, save_dir=args.default_root_dir
        )
    else:
        logger = CSVLogger(args.default_root_dir)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    num_gpus = torch.cuda.device_count()
    devices = None if num_gpus == 0 else num_gpus
    accelerator = "cpu" if num_gpus == 0 else "gpu"

    trainer = pl.Trainer(
        max_epochs=-1,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        deterministic=True,
        default_root_dir=args.default_root_dir,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    )

    model = EngawaModel(
        tokenizer,
        args.lr,
        args.weight_decay,
        args.num_warmup_steps,
        args.max_steps if args.max_steps > -1 else 500000,
        size=args.model_size,
    )

    if args.ckpt_path is not None:
        print(f"Resume training from {args.ckpt_path}...")
    trainer.fit(model, train_dl, val_dl, ckpt_path=args.ckpt_path)
