"""
One-shot offline training on a pre-scored CSV file.

CSV expected columns: prompt, sequence, reward
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from accelerate.utils import set_seed

from src.ProtRL_Trainer import ProtRLTrainingArgument
from src.pLM_GRPO import ProtRL_GRPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--output", type=str, default="./output_exp")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--split_percent", type=float, default=0.2)
parser.add_argument("--ref_model", type=str, default=None)

args = parser.parse_args()
ref_model = args.ref_model


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


seed_everything(42)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_dir,
    add_eos_token=True,
    add_bos_token=False,
    use_fast=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_sequence(sequence):
    return str(sequence).replace(" ", "")


def build_dataset():
    df = pd.read_csv(args.csv)
    rows = [
        {"prompt": row["prompt"], "completion": format_sequence(row["sequence"]), "reward": float(row["reward"])}
        for _, row in df.iterrows()
    ]
    return Dataset.from_list(rows)


dataset = build_dataset()
split = dataset.train_test_split(test_size=args.split_percent, seed=42, shuffle=True)

training_args = ProtRLTrainingArgument(
    output_dir=args.output,
    logging_steps=100,
    beta=args.beta,
    num_train_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
)

trainer = ProtRL_GRPOTrainer(
    model=args.model_dir,
    ref_model=ref_model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model()
torch.cuda.empty_cache()
