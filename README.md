# engawa

**NOT YET FULLY TESTED**

A simple implementation to pre-train BART from scratch with your own corpus.


# Usage

Soon, I will make this pip-installable with CLI commands but at the moment, you need to run it as a repository.

## Installation

```bash
git clone git@github.com:sobamchan/engawa.git && cd engawa
poetry install
```

## Build tokenizer

```bash
python engawa/tokenizer.py --data-path /path/to/train.txt --save-dir /path/to/save

# Checkout other options by
python engawa/tokenizer.py -h
```

## Pre-train BART

```bash
python engawa/train.py --tokenizer-file /path/to/tokenizer.json --train-file /path/to/train.txt --val-file /path/to/val.txt --default-root-dir /path/to/save/things

# Checkout other options by
python engawa/train.py -h
```
