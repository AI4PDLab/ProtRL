import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from accelerate.utils import set_seed

from src.ProtRL_Trainer import ProtRLTrainingArgument, checkpoint_load, load_optimizer_scheduler, save_config
from src.pLM_GRPO import ProtRL_GRPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--max_iteration_num", type=int, required=True)
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save results")

args = parser.parse_args()

CONFIG = {
    "beta": 0.01,
    "seed": 42,
    "learning_rate": 1e-5,
    "num_epochs": 1,
    "split_percent": 0.2,
    "adam_betas": [0.9, 0.98],
    "epsilon": 1e-8,
    "adam_decay": 0.1,
}

os.makedirs(args.output_dir, exist_ok=True)
CONFIG.update({
    "iteration_num": args.iteration_num,
    "label": args.label,
    "model_dir": args.model_dir,
    "max_iteration_num": args.max_iteration_num,
    "output_dir": args.output_dir,
})
save_config(CONFIG, os.path.join(args.output_dir, f"config_iteration{args.iteration_num}.yaml"))


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


seed_everything(CONFIG["seed"])

tokenizer = AutoTokenizer.from_pretrained(
    args.model_dir,
    add_eos_token=True,
    add_bos_token=False,
    use_fast=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



def format_sequence(sequence, tokenizer):
    return str(sequence).replace(" ", "")



def generate_dataset(iteration_num, label):
    logs_path = os.path.join(args.output_dir, "logs.csv")
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"logs.csv not found in {args.output_dir}")
    df = pd.read_csv(logs_path)
    df = df[df["iteration_num"] == iteration_num]

    def build_completion(seq):
        # The stored sequence includes the prompt (generation is prompted with [BOS]+label,
        # and the label - e.g. leading "M" - ends up in the decoded sequence). The trainer
        # re-adds [BOS]+prompt, so strip the prompt here to avoid duplicating it and to make
        # the training tokens exactly match what the model generated: [BOS, M, <rest>, EOS].
        seq = format_sequence(seq, tokenizer)
        if label and seq.startswith(label):
            seq = seq[len(label):]
        return seq

    rows = [
        # reward is on the FULL generated length (what we actually want near 20),
        # while the completion is the part after the prompt.
        {"prompt": label, "completion": build_completion(row["sequence"]), "reward": float(-abs(20 - len(row["sequence"])))}
        for _, row in df.iterrows()
    ]
    return Dataset.from_list(rows)


dataset = generate_dataset(args.iteration_num, args.label)
split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)
train_dataset = split["train"]
eval_dataset = split["test"]

if args.iteration_num > 1:
    prev_model_dir = os.path.join(args.output_dir, f"output_iteration{args.iteration_num - 1}")
    model = prev_model_dir if os.path.exists(prev_model_dir) else args.model_dir
    checkpoint = checkpoint_load(model)
else:
    model = args.model_dir
    checkpoint = None

lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)
current_lr = lr_list[args.iteration_num - 1].item()
model_obj, optimizer, scheduler = load_optimizer_scheduler(model, checkpoint, current_lr, CONFIG)

output_model_dir = os.path.join(args.output_dir, f"output_iteration{args.iteration_num}")
training_args = ProtRLTrainingArgument(
    output_dir=output_model_dir,
    logging_steps=100,
    beta=CONFIG["beta"],
    num_train_epochs=CONFIG["num_epochs"],
    learning_rate=current_lr,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="steps",
    save_total_limit=1,
    save_steps=5,
)

print(f"Method: pLM_GRPO, Model: {model}")

trainer = ProtRL_GRPOTrainer(
    model=model_obj,
    ref_model=args.model_dir,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, scheduler),
)

trainer.train()
trainer.save_model()
torch.cuda.empty_cache()
