import click

from engawa.tokenizer import train_tokenizer
from engawa.train import train_model


@click.group()
def cli():
    pass


cli.add_command(train_tokenizer)
cli.add_command(train_model)


def main():
    cli()
