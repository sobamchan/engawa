import nltk
from datasets.load import load_dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer
from torch.utils.data import DataLoader
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast

from engawa.data_collator import DataCollatorForDenoisingTasks


def tokenize_sents(
    sentence_tokenizer: PunktSentenceTokenizer,
    text: str,
    bos_token: str,
    eos_token: str,
) -> str:
    return f"{eos_token}{bos_token}".join(sentence_tokenizer.tokenize(text))


def get_dataloader(
    data_path: str,
    split: str,
    bs: int,
    tokenizer: BartTokenizerFast,
    mask_ratio: float,
    poisson_lambda: float,
    max_length: int,
) -> DataLoader:
    nltk.download("punkt")
    sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    ds = (
        load_dataset("text", data_files=data_path, split="train")
        .map(
            lambda x: tokenizer(
                tokenize_sents(sentence_tokenizer, x["text"], bos_token, eos_token),
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        )
        .remove_columns(["text"])
    )

    assert isinstance(tokenizer.bos_token_id, int)
    dc = DataCollatorForDenoisingTasks(
        tokenizer=tokenizer, mask_ratio=mask_ratio, poisson_lambda=poisson_lambda
    )

    shuffle = True if split == "train" else False
    return DataLoader(ds, batch_size=bs, collate_fn=dc, shuffle=shuffle)
