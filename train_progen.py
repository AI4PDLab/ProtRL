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
from src.utils import checkpoint_load, save_config
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

# ProGen import
try:
    from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM
except ImportError:
    print("Error: Could not import ProGenForCausalLM. Ensure 'progen' directory is in the path.")
    # Fallback or exit?
    # sys.exit(1) 
    pass

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
dataset = generate_dataset(args.iteration_num, args.label)
split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)

train_dataset = split['train']
eval_dataset = split['test'] 

from transformers import PreTrainedTokenizerFast

# Load tokenizer
tokenizer_dir = args.model_dir
tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")

# If not found, check parent directories (up to 2 levels)
if not os.path.exists(tokenizer_file):
    parent = os.path.dirname(tokenizer_dir)
    if os.path.exists(os.path.join(parent, "tokenizer.json")):
        tokenizer_file = os.path.join(parent, "tokenizer.json")
    else:
        grandparent = os.path.dirname(parent)
        if os.path.exists(os.path.join(grandparent, "tokenizer.json")):
            tokenizer_file = os.path.join(grandparent, "tokenizer.json")

if os.path.exists(tokenizer_file):
    print(f"Loading tokenizer from {tokenizer_file}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    # Ensure special tokens are set if needed, though PreTrainedTokenizerFast usually handles them from json
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
else:
    print(f"Warning: tokenizer.json not found in {tokenizer_dir} or parents. Falling back to AutoTokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

# Determine model and checkpoint
if args.iteration_num > 1: 
    # Previous model should be in the output directory
    prev_model_dir = os.path.join(args.output_dir, f"output_iteration{args.iteration_num-1}")
    if os.path.exists(prev_model_dir):
        model_path = prev_model_dir
        checkpoint = checkpoint_load(prev_model_dir) 
    else:
        model_path = f"output_iteration{args.iteration_num-1}"
        checkpoint = checkpoint_load(f"output_iteration{args.iteration_num-1}")
else:
    model_path = args.model_dir
    checkpoint = None
 
# Learning rate scheduler setup
lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)
current_lr = lr_list[args.iteration_num-1].item()

# Load Model (ProGen)
print(f"Loading ProGen model from {model_path}")
model = ProGenForCausalLM.from_pretrained(model_path)
ref_model = ProGenForCausalLM.from_pretrained(model_path)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr = current_lr,
    betas = CONFIG["adam_betas"],
    eps = CONFIG["epsilon"],
    weight_decay = CONFIG["adam_decay"]
)

# Load Optimizer State if Checkpoint Exists
if checkpoint is not None:
    optim_state_path = checkpoint / "optimizer.pt"
    if optim_state_path.exists():
        print(f"Loading optimizer state from {optim_state_path}")
        saved_optim_state = torch.load(optim_state_path, map_location="cpu")
        optimizer.load_state_dict(saved_optim_state)
        for group in optimizer.param_groups:
                group["lr"] = current_lr
                group["initial_lr"] = current_lr

# Scheduler
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

# Training arguments
output_model_dir = os.path.join(args.output_dir, f"output_iteration{args.iteration_num}")
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
    save_steps=5,
    num_generations=8, 
    importance_sampling_level="sequence"
)

print("Model:", model)

# Initialize trainer
trainer = pLM_GRPOTrainer(
    model=model,
    ref_model=ref_model,
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
