import pytorch_lightning as pl
from transformers.models.bart.modeling_bart import (
    BartConfig, BartForConditionalGeneration)
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from transformers.optimization import (
    AdamW, get_polynomial_decay_schedule_with_warmup)


class EngawaModel(pl.LightningModule):
    def __init__(
        self,
        tokenizer: BartTokenizerFast,
        lr: float,
        weight_decay: float,
        num_warmup_steps: int,
        num_training_steps: int,
        model_type: str,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        eos_token_id = tokenizer.eos_token_id
        bos_token_id = tokenizer.bos_token_id
        pad_token_id = tokenizer.pad_token_id
        vocab_size = tokenizer.vocab_size
        config = BartConfig.from_pretrained(f"facebook/bart-{model_type}")
        config.update_from_string(
            f"eos_token_id={eos_token_id},bos_token_id={bos_token_id},pad_token_id={pad_token_id},vocab_size={vocab_size}"
        )

        self.bart = BartForConditionalGeneration(config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        out = self.bart(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
        )
        loss = out.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.bart(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
        )
        loss = out.loss
        self.log("val_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        bart_params = [
            {
                "params": [
                    p
                    for n, p in self.bart.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.bart.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(bart_params, lr=self.lr)

        return (
            [optimizer],
            [
                get_polynomial_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.num_training_steps,
                )
            ],
        )
