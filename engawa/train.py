from argparse import ArgumentParser

import pytorch_lightning as pl
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
    parser.add_argument("--wandb-proj-name", type=str, required=False, default=None)

    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--num-training-steps", type=int, default=500000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-warmup-steps", type=int, default=10000)

    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--poisson_lambda", type=float, default=3.5)
    args = parser.parse_args()

    tokenizer = BartTokenizerFast(tokenizer_file=args.tokenizer_file)

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
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        deterministic=True,
        default_root_dir=args.default_root_dir,
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    model = EngawaModel(
        tokenizer,
        args.lr,
        args.weight_decay,
        args.num_warmup_steps,
        args.num_training_steps,
    )
    trainer.fit(model, train_dl, val_dl)