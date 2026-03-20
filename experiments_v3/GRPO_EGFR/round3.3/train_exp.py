
from src.utils import *
from src.pLM_weigtedDPO import weighted_DPO
from src.pLM_GRPO import pLM_GRPOTrainer

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel)
from trl.trainer.utils import pad
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import argparse
import torch
import numpy as np
import random
import pandas as pd
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--output", type=str, default="./output_")
parser.add_argument("--learning_rate", type=float, default=2e-6)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument("--split_percent", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--ref_model", type=str, required=False)

args = parser.parse_args()

if args.ref_model is None:
    ref_model = args.model_dir
else:
    ref_model = args.ref_model

CONFIG = {
        "learning_rate":  args.learning_rate,
        "num_epochs":     args.num_epochs,
        "split_percent":  args.split_percent,
        "beta":           args.beta,
        "seed":           42,
        }

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    set_seed(seed)


def reward_len(completions, **kwargs):
    return 0


def generate_dataset():

    df = pd.read_csv(args.csv)
    
    rows = []
    for idx, entry in df.iterrows():
        sequence = entry["sequence"]
        advantage = entry["advantage"]
        prompt = entry["prompt"]
        
        rows.append({
            "prompt": prompt,
            "completion": sequence,
            "reward": advantage
        })
    
    return Dataset.from_list(rows)


seed_everything(CONFIG["seed"])

# create dataset
dataset = generate_dataset()
split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)

train_dataset = split['train']
eval_dataset   = split['test'] 

tokenizer = AutoTokenizer.from_pretrained(ref_model,
                                          add_eos_token=False, # NEED this for training NOT for generate() else add eos at the end of promt
                                          add_bos_token=False,
                                          use_fast=True)


training_args = GRPOConfig(output_dir=args.output, 
                           logging_steps=100,
                           beta=CONFIG["beta"],
                           num_train_epochs = CONFIG["num_epochs"],
                           learning_rate = CONFIG["learning_rate"],
                           do_train = True, 
                           do_eval = True, 
                           eval_strategy = "epoch",
                           save_strategy = "steps",                     
                           eval_steps = 500, 
                           save_total_limit = 1,
                           save_steps = 5,
                           num_generations = 8)


trainer = pLM_GRPOTrainer(
    model= args.model_dir,
    ref_model = ref_model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer)

trainer.train()
trainer.save_model()
