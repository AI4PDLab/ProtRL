import os
import sys
import argparse
import random
import math
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from accelerate.utils import set_seed
from trl import GRPOConfig, GRPOTrainer
from pathlib import Path

# Ensure project root is on sys.path
here = Path(__file__).resolve()
for parent in (here.parent, *here.parents):
    if (parent / "src" / "__init__.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from src.pLM_GRPO import pLM_GRPOTrainer
from src.pLM_rankedDPO import weighted_DPO
from src.utils import checkpoint_load, load_optimizer_scheduler, save_config

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--max_iteration_num", type=int, required=True)
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results for this specific run")
parser.add_argument("--method", type=str, choices=["pLM_GRPO", "weighted_DPO"], required=True)
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
    """Sets seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)

def reward_len(completions, **kwargs):
    """Reward function: penalize deviation from length 50."""
    # TRL GRPOTrainer expects a list of rewards (one per completion)
    # But pLM_GRPO/weighted_DPO might handle it differently?
    # pLM_GRPO uses implicit reward from logs.csv usually, but here we might need explicit reward for trl_GRPO?
    # Wait, pLM_GRPO and weighted_DPO in this codebase use the 'rewards' column from the dataset (which comes from logs.csv).
    # trl_GRPO usually calculates rewards on the fly using reward_funcs.
    # The user said "trl GRPO with the same training objective of reducing the lenght of the sequence up to 50 characters".
    # So for trl_GRPO, we need a callable reward function.
    rewards = []
    for c in completions:
        rewards.append(float(-abs(50 - len(c))))
    return rewards

def format_sequence(sequence, label):
    """Formats the sequence (currently returns as is)."""
    return sequence

def generate_dataset(iteration_num, label):
    """Generates dataset from logs.csv."""
    logs_path = os.path.join(args.output_dir, "logs.csv")
    
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"logs.csv not found in {args.output_dir}")
        
    df = pd.read_csv(logs_path)
    df = df[df["iteration_num"] == iteration_num]
    
    rows = []
    for idx, entry in df.iterrows():
        sequence = entry["sequence"]
       
        rows.append({
            "prompt": label,
            "completion": format_sequence(sequence, label),
            "reward": float(-abs(50-len(sequence)))
        })
    
    return Dataset.from_list(rows)

# Set seed
seed_everything(CONFIG["seed"])

# Create dataset
dataset = generate_dataset(args.iteration_num, args.label)
split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)

train_dataset = split['train']
eval_dataset = split['test'] 

# Load tokenizer
tokenizer_dir = args.model_dir
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_dir,
    add_eos_token=True, 
    add_bos_token=False,
    use_fast=True
)

# Determine model and checkpoint
if args.iteration_num > 1: 
    prev_model_dir = os.path.join(args.output_dir, f"output_iteration{args.iteration_num-1}")
    if os.path.exists(prev_model_dir):
        model = prev_model_dir
        checkpoint = checkpoint_load(prev_model_dir)
    else:
        # Fallback (shouldn't happen in clean benchmark)
        model = f"output_iteration{args.iteration_num-1}"
        checkpoint = None
else:
    model = args.model_dir
    checkpoint = None
 
# Learning rate scheduler setup
lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)

# Load optimizer and scheduler
model_obj, optimizer, scheduler = load_optimizer_scheduler(
    model, 
    checkpoint, 
    lr_list[args.iteration_num-1].item(), 
    CONFIG
)

# Training arguments
output_model_dir = os.path.join(args.output_dir, f"output_iteration{args.iteration_num}")
training_args = GRPOConfig(
    output_dir=output_model_dir, 
    logging_steps=100,
    beta=CONFIG["beta"],
    num_train_epochs=CONFIG["num_epochs"],
    learning_rate=lr_list[args.iteration_num-1].item(),
    do_train=True, 
    do_eval=True, 
    eval_strategy="epoch",
    save_strategy="steps",                     
    eval_steps=500, 
    save_total_limit=1,
    save_steps=5,
    num_generations=8, 
    importance_sampling_level=CONFIG["importance_sampling"]
)

print(f"Method: {args.method}, Model: {model}")

# Initialize trainer based on method
if args.method == "pLM_GRPO":
    trainer = pLM_GRPOTrainer(
        model=model_obj,
        ref_model=args.model_dir, 
        reward_funcs=reward_len, # Dummy for pLM_GRPO as it uses dataset rewards
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler)
    )
elif args.method == "weighted_DPO":
    trainer = weighted_DPO(
        model=model_obj,
        ref_model=args.model_dir,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler)
    )

    # Custom trainers might need this manual assignment if not handled in super
    trainer.lr_scheduler = scheduler
    trainer.lr_scheduler_state = None

# Train and save
trainer.train()
trainer.save_model()

torch.cuda.empty_cache()
