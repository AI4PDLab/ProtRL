import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from accelerate.utils import set_seed
from trl import GRPOConfig
from src.pLM_GRPO import pLM_GRPOTrainer
from src.utils import checkpoint_load, load_optimizer_scheduler, save_config

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--max_iteration_num", type=int, required=True)
parser.add_argument("--output_dir", type=str, default=".")
args = parser.parse_args()

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

os.makedirs(args.output_dir, exist_ok=True)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)

def reward_func(completions, **kwargs):
    return 0

def generate_dataset(iteration_num, label, output_dir):
    logs_path = os.path.join(output_dir, "logs.csv")
    if not os.path.exists(logs_path):
        return Dataset.from_dict({"prompt": [], "completion": [], "reward": []})
        
    df = pd.read_csv(logs_path)
    df = df[df["iteration_num"] == iteration_num]
    
    rows = []
    for _, entry in df.iterrows():
        rows.append({
            "prompt": label,
            "completion": entry["sequence"],
            "reward": float(entry.get("clean_reward", 0.0))
        })
    return Dataset.from_list(rows)

seed_everything(CONFIG["seed"])

dataset = generate_dataset(args.iteration_num, args.label, args.output_dir)
if len(dataset) == 0:
    exit(0)

split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, add_eos_token=False, add_bos_token=False, use_fast=True)
tokenizer.eos_token_id = 1
tokenizer.pad_token_id = 0

if args.iteration_num > 1: 
    prev_model = os.path.join(args.output_dir, f"output_iteration{args.iteration_num-1}")
    if os.path.exists(prev_model):
        model = prev_model
        checkpoint = checkpoint_load(prev_model)
    else:
        model = args.model_dir
        checkpoint = None
else:
    model = args.model_dir
    checkpoint = None
 
lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)
model, optimizer, scheduler = load_optimizer_scheduler(model, checkpoint, lr_list[args.iteration_num-1].item(), CONFIG)

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

trainer = pLM_GRPOTrainer(
    model=model,
    ref_model=args.model_dir,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=split['train'],
    eval_dataset=split['test'],
    processing_class=tokenizer,
    optimizers=(optimizer, scheduler)
)

trainer.lr_scheduler = scheduler
trainer.train()
trainer.save_model()
torch.cuda.empty_cache()
