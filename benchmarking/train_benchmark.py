import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate.utils import set_seed
from trl import GRPOConfig, GRPOTrainer
from pathlib import Path

# Project root is one level above benchmarking/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ProtRL_Trainer import ProtRLTrainingArgument, checkpoint_load, load_optimizer_scheduler, save_config
from src.pLM_GRPO import ProtRL_GRPOTrainer
from src.pLM_weightedDPO import ProtRL_wDPOTrainer, ProtRL_wDPOTrainingArgument
from src.pLM_REINFORCE import ProtRL_REINFORCETrainer, ProtRL_REINFORCETrainingArgument

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--max_iteration_num", type=int, required=True)
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results for this specific run")
parser.add_argument("--method", type=str, choices=["ProtRL_GRPO", "ProtRL_wDPO", "trl_GRPO", "ProtRL_REINFORCE"], required=True)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--importance_sampling", type=str, default="sequence", choices=["token", "sequence"])

args = parser.parse_args()

# Configuration
CONFIG = {
    "method": args.method,
    "beta": args.beta,
    "seed": 42,
    "learning_rate": args.learning_rate,
    "batch_size": 15,
    "num_epochs": 1,
    "split_percent": 0.2,
    "adam_betas": [0.9, 0.98],
    "epsilon": 1e-8,
    "adam_decay": 0.1,
    "importance_sampling": args.importance_sampling
}

# Save configuration
os.makedirs(args.output_dir, exist_ok=True)
CONFIG.update({
    "iteration_num": args.iteration_num,
    "label": args.label,
    "model_dir": args.model_dir,
    "max_iteration_num": args.max_iteration_num,
    "output_dir": args.output_dir
})
save_config(CONFIG, os.path.join(args.output_dir, f"config_iteration{args.iteration_num}.yaml"))

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)

generation_log = []

def reward_len(completions, **kwargs):
    """Reward function for trl_GRPO: penalize deviation from length 50.

    Also logs every generated completion (trl doesn't otherwise persist
    them), so trl_GRPO runs end up with a logs.csv like the offline methods.
    """
    trainer_state = kwargs.get("trainer_state")
    step = trainer_state.global_step if trainer_state is not None else 0
    for i, completion in enumerate(completions):
        sequence = completion.replace(" ", "")
        generation_log.append({
            "name": f"step{step}_{i}",
            "sequence": sequence,
            "length": len(sequence),
            "iteration_num": step,
        })
    return [float(-abs(100 - len(c.replace(" ", "")))) for c in completions]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_dir,
    add_eos_token=True,
    add_bos_token=False,
    use_fast=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

seed_everything(CONFIG["seed"])


def format_sequence(sequence, tokenizer):
    name = getattr(tokenizer, "name_or_path", "") or ""
    if "protgpt3" in name.lower():
        if " " not in sequence:
            return " ".join(list(sequence))
    return sequence


def generate_dataset(iteration_num, label):
    """Generates dataset from logs.csv (used by offline methods)."""
    logs_path = os.path.join(args.output_dir, "logs.csv")

    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"logs.csv not found in {args.output_dir}")

    df = pd.read_csv(logs_path)
    df = df[df["iteration_num"] == iteration_num]

    rows = [
        {
            "prompt": label,
            "completion": format_sequence(entry["sequence"], tokenizer),
            "reward": float(-abs(100 - len(entry["sequence"]))),
        }
        for _, entry in df.iterrows()
    ]
    return Dataset.from_list(rows)

def generate_dummy_dataset(num_samples=500):
    """Generates dummy dataset for trl_GRPO (online)."""
    return Dataset.from_list([{"prompt": args.label, "completion": ""} for _ in range(num_samples)])


# Create dataset
if args.method == "trl_GRPO":
    dataset = generate_dummy_dataset()
else:
    dataset = generate_dataset(args.iteration_num, args.label)

split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)
train_dataset = split["train"]
eval_dataset = split["test"]

# Determine model and checkpoint
if args.iteration_num > 1:
    prev_model_dir = os.path.join(args.output_dir, f"output_iteration{args.iteration_num - 1}")
    model = prev_model_dir if os.path.exists(prev_model_dir) else args.model_dir
    checkpoint = checkpoint_load(model)
else:
    model = args.model_dir
    checkpoint = None

# Learning rate scheduler setup
lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)
current_lr = lr_list[args.iteration_num - 1].item()
model_obj, optimizer, scheduler = load_optimizer_scheduler(model, checkpoint, current_lr, CONFIG)

# Training arguments
output_model_dir = os.path.join(args.output_dir, f"output_iteration{args.iteration_num}")

print(f"Method: {args.method}, Model: {model}")

if args.method == "trl_GRPO":
    training_args = GRPOConfig(
        output_dir=output_model_dir,
        logging_steps=100,
        beta=CONFIG["beta"],
        num_train_epochs=CONFIG["num_epochs"],
        learning_rate=current_lr,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="steps",
        eval_steps=500,
        save_total_limit=1,
        save_steps=500,
        max_steps=CONFIG["max_iteration_num"],
        num_generations=32,
        importance_sampling_level=CONFIG["importance_sampling"],
    )
else:
    training_args_cls = {
        "ProtRL_GRPO": ProtRLTrainingArgument,
        "ProtRL_wDPO": ProtRL_wDPOTrainingArgument,
        "ProtRL_REINFORCE": ProtRL_REINFORCETrainingArgument,
    }[args.method]
    training_args = training_args_cls(
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

# Initialize trainer
if args.method == "ProtRL_GRPO":
    trainer = ProtRL_GRPOTrainer(
        model=model_obj,
        ref_model=args.model_dir,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
    )
elif args.method == "ProtRL_wDPO":
    trainer = ProtRL_wDPOTrainer(
        model=model_obj,
        ref_model=args.model_dir,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
    )
elif args.method == "trl_GRPO":
    trainer = GRPOTrainer(
        model=model_obj,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
    )
elif args.method == "ProtRL_REINFORCE":
    trainer = ProtRL_REINFORCETrainer(
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

if args.method == "trl_GRPO":
    pd.DataFrame(generation_log).to_csv(os.path.join(args.output_dir, "logs.csv"), index=False)

torch.cuda.empty_cache()