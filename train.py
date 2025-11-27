import os
import argparse
import random
import math
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from accelerate.utils import set_seed
from trl import GRPOConfig
from src.pLM_GRPO import pLM_GRPOTrainer
from src.pLM_rankedDPO import weighted_DPO
# Local imports
from src.utils import checkpoint_load, load_optimizer_scheduler, save_config

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--max_iteration_num", type=int, required=True)
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save results")

args = parser.parse_args()

# Configuration
CONFIG = {
    "beta": 0.01,
    "seed": 42,
    "learning_rate": 1e-6,
    "batch_size": 15,
    "num_epochs": 1,
    "split_percent": 0.2,
    "adam_betas": [0.9, 0.98],
    "epsilon": 1e-8,
    "adam_decay": 0.1,
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
    """Dummy reward function (returns 0)."""
    return 0

def format_sequence(sequence, label):
    """Formats the sequence (currently returns as is)."""
    return sequence

def generate_dataset(iteration_num, label):
    """Generates dataset from logs.csv."""
    # Assuming logs.csv is in the current directory or we need to look in output_dir?
    # For now, keeping it as is, but ideally it should be in output_dir if generated there.
    # Checking both locations
    logs_path = "logs.csv"
    if not os.path.exists(logs_path) and os.path.exists(os.path.join(args.output_dir, "logs.csv")):
        logs_path = os.path.join(args.output_dir, "logs.csv")
        
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
# Note: root_dir and fasta_file variables were unused in original code, removed for cleanliness
dataset = generate_dataset(args.iteration_num, args.label)
split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)

train_dataset = split['train']
eval_dataset = split['test'] 

# Load tokenizer
tokenizer_dir = args.model_dir
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_dir,
    add_eos_token=True, # Needed for training
    add_bos_token=False,
    use_fast=True
)

# Determine model and checkpoint
if args.iteration_num > 1: 
    # Previous model should be in the output directory
    prev_model_dir = os.path.join(args.output_dir, f"output_iteration{args.iteration_num-1}")
    if os.path.exists(prev_model_dir):
        model = prev_model_dir
        checkpoint = checkpoint_load(prev_model_dir) # Assuming checkpoint_load handles full path
    else:
        # Fallback to local dir if not found (legacy behavior)
        model = f"output_iteration{args.iteration_num-1}"
        checkpoint = checkpoint_load(f"output_iteration{args.iteration_num-1}")
else:
    model = args.model_dir
    checkpoint = None
 
# Learning rate scheduler setup
lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)

# Load optimizer and scheduler
model, optimizer, scheduler = load_optimizer_scheduler(
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
    importance_sampling_level="sequence"
)

print("Model:", model)

# Initialize trainer
trainer = pLM_GRPOTrainer(
    model=model,
    ref_model=args.model_dir,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, scheduler)
)

trainer.lr_scheduler = scheduler
trainer.lr_scheduler_state = None

# Train and save
trainer.train()
trainer.save_model()

torch.cuda.empty_cache()
