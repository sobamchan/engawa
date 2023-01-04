from argparse import ArgumentParser
import os

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


def main(data_path: str, save_dir: str, batch_size: int):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Set the training hyperparameters
    trainer = trainers.BpeTrainer(
        vocab_size=50000,
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        required=True,
        help="Path to a training text file.",
    )
    parser.add_argument(
        "--save-dir",
        "-s",
        type=str,
        required=True,
        help="Dir to save trained tokenizer.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    main(args.data_path, args.save_dir, args.batch_size)
