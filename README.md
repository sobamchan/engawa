# engawa

<img align="center" src="img/logo.jpg" width="200" height="200" />

**NOT YET FULLY TESTED**

A simple implementation to pre-train BART from scratch with your own corpus.


# Usage

Soon, I will make this pip-installable with CLI commands but at the moment, you need to run it as a repository.

## Installation

```bash
pip install engawa
```

## Build tokenizer

```bash
engawa train-tokenizer --data-path /path/to/train.txt --save-dir /path/to/save

# Checkout other options by
engawa train-tokenizer --help
```

## Pre-train BART

```bash
engawa train-model \
  --tokenizer-file /path/to/tokenizer.json \
  --train-file /path/to/train.txt \
  --val-file /path/to/val.txt \
  --default-root-dir /path/to/save/things

# Checkout other options by
engawa train-model --help
```
