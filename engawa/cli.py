import typer

from engawa.tokenizer import train_tokenizer
from engawa.train import train_model

app = typer.Typer()
app.command()(train_tokenizer)
app.command()(train_model)
