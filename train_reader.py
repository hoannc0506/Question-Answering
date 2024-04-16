import numpy as np
from tqdm.auto import tqdm
import collections
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
from data_utils import SquadDataProcessor
from config import QAConfig
import evaluate

cfg = QAConfig()

# init model
model = AutoModelForQuestionAnswering.from_pretrained(cfg.MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

# define train params
args = TrainingArguments(
    output_dir=cfg.ckpt_dir,
    evaluation_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=cfg.train_batch_size,
    per_device_eval_batch_size=cfg.eval_batch_size,
    save_total_limit=cfg.save_total_limit,
    num_train_epochs=cfg.num_train_epochs,
    learning_rate=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
    fp16=cfg.fp16,
    report_to="wandb", # wandb logging
    run_name="qa-squadv2-distilbert" # wandb run name
)

if __name__ == "__main__":
    # load data
    raw_train_dataset = load_dataset(cfg.DATASET_NAME, split ="train", num_proc=cfg.NUM_PROC)
    raw_val_dataset = load_dataset(cfg.DATASET_NAME, split ="validation", num_proc=cfg.NUM_PROC)
    
    # tokenize data
    data_processor = SquadDataProcessor(cfg)
    train_data = data_processor.process_data(raw_train_dataset, data_type="train")
    val_data = data_processor.process_data(raw_val_dataset, data_type="validation")

    # init trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer
    )

    # finetuned model
    trainer.train()
