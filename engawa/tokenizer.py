import os

import click
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
PAD = "<pad>"
MASK = "<mask>"


def text_iterator(docs: list, bs: int):
    for i in range(0, len(docs), bs):
        yield docs[i : i + bs]
    pass


def read_iterator(dpath: str, bs: int):
    with open(dpath, "r") as f:
        text_batch = []
        cnt = 0
        for line in f:
            if cnt == bs:
                yield text_batch
                text_batch = []
                cnt = 0
            text_batch.append(line)
            cnt += 1


@click.command()
@click.option(
    "--data-path", type=str, required=True, help="Path to a training text file."
)
@click.option(
    "--save-dir", type=str, required=True, help="Dir to save trianed tokenizer."
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for building tokenizer, smaller value for limited resourced situation but takes more time.",
)
@click.option(
    "--vocab-size",
    type=int,
    default=50000,
    help="Number of unique tokens in the vocabulary.",
)
def train_tokenizer(data_path: str, save_dir: str, batch_size: int, vocab_size: int):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Set the training hyperparameters
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[BOS, PAD, EOS, UNK, MASK],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train it with either files or an iterator:
    tokenizer.train_from_iterator(read_iterator(data_path, batch_size), trainer=trainer)

    # Attach the Roberta Processing
    tokenizer.post_processor = processors.RobertaProcessing(
        sep=(EOS, tokenizer.token_to_id(EOS)), cls=(BOS, tokenizer.token_to_id(BOS))
    )

    # And save it:
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
