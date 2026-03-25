#
# ----- MINIMAL LORA IMPLEMENTATION -----
#
# The original code has been minimally modified to incorporate LoRA finetuning.
# Key changes are marked with comments.

from src.utils import *
from src.pLM_weigtedDPO import weighted_DPO
from src.pLM_rankedDPO import ranked_DPO
from src.pLM_GRPO import pLM_GRPOTrainer

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers.utils.import_utils import is_rich_available
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    get_linear_schedule_with_warmup # Added for scheduler
)
from peft import LoraConfig, get_peft_model, PeftModel
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
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--max_iteration_num", type=int, required=True)
parser.add_argument("--d_path", type=str, required=False)

args = parser.parse_args()


CONFIG = {
    "beta": 0.1,
    "seed": 42,
    "learning_rate": 1e-4,
    "num_epochs": 1,
    "split_percent": 0.2,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "adam_decay": 0.1,
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


def format_sequence(sequence, label):
    return f"<|bos|>{sequence}<|eos|>"

def generate_dataset(iteration_num, label):

    df_norm = pd.read_csv(f"{args.d_path}logs_output.csv", on_bad_lines='skip')
    print(df_norm)
    df_norm = df_norm[df_norm["iteration_num"] == iteration_num-1]
    df_norm = df_norm.drop_duplicates(subset=['sequence'])
    print(df_norm)
    # Existing normalizations remain unchanged
    df_norm['length_norm']              = abs(140 - df_norm['lenght'])
    df_norm['b_plddt']                  = df_norm['b_plddt']              /  df_norm['b_plddt'].max()
    df_norm['i_plddt']                  = df_norm['i_plddt']              / df_norm['i_plddt'].max()
    df_norm['i_ptm']                    = df_norm['i_ptm']                / df_norm['i_ptm'].max()
    df_norm['ipsae']                    = df_norm['ipsae']
    df_norm['b_pae']                    = df_norm['b_pae']                / df_norm['b_pae'].max()
    df_norm['pMPNN']                    = df_norm['pMPNN']                / df_norm['pMPNN'].max()
    df_norm['shape_complimentary']      = df_norm['shape_complimentary']  / df_norm['shape_complimentary'].max()
    df_norm['hydrophobicity']           = df_norm['hydrophobicity']       / df_norm['hydrophobicity'].max()
    df_norm['REU']                      = df_norm['REU']                  / df_norm['REU'].max()
    df_norm['interface_dSASA']          = df_norm['interface_dSASA']      / df_norm['interface_dSASA'].max()
    df_norm['uns_hydrogens']            = abs(0 - df_norm['uns_hydrogens'])
    df_norm['contact_probs']            = df_norm['contact_probs']        / df_norm['contact_probs'].max()
    df_norm['d_rmsd']                   = df_norm['d_rmsd']               / df_norm['d_rmsd'].max()
    df_norm['binder_sasa']              = df_norm['binder_sasa']          / df_norm['binder_sasa'].max()
    #df_norm['delta_delta_interaction']  = df_norm['delta_delta_interaction'] / df_norm['delta_delta_interaction'].max()
    df_norm['helicity']                 = df_norm['helicity']              / df_norm['helicity'].max()
    df_norm['i_pae']                    = df_norm['i_pae']                  / df_norm['i_pae'].max()
    df_norm['lis']         = df_norm['lis']        / df_norm['lis'].max()
    df_norm['d0res']       = df_norm['d0res']      / df_norm['d0res'].max()
    df_norm['d0chn']       = df_norm['d0chn']      / df_norm['d0chn'].max()
    df_norm['d0dom']       = df_norm['d0dom']      / df_norm['d0dom'].max()
    df_norm['iptm_d0chn']  = df_norm['iptm_d0chn'] / df_norm['iptm_d0chn'].max()
    df_norm['pdockq2']     = df_norm['pdockq2']    / df_norm['pdockq2'].max()
    df_norm['pdockq']      = df_norm['pdockq']     / df_norm['pdockq'].max()
    #df_norm['iptm_af']     = df_norm['iptm_af']    / df_norm['iptm_af'].max()
    df_norm['ptm']          = df_norm['ptm']       /df_norm['ptm'].max()
    df_norm['t_pae']        = df_norm['t_pae'] /    df_norm['t_pae'].max()

        
    df_norm["score"] = (
            #+ df_norm['i_plddt']
            - 0.1 * df_norm["length_norm"]
            #- 0.8 * df_norm['REU']
            #+ 0.8 * df_norm["b_plddt"]
            #+ 0.8 * df_norm["i_ptm"]
            - df_norm["b_pae"]
            - df_norm['i_pae']
            #- 0.5 * df_norm["pMPNN"]
            + 0.3 * df_norm["shape_complimentary"]
            #- 0.4 * df_norm["hydrophobicity"]
            + 0.7 * df_norm["interface_dSASA"]
            #- 0.7 * df_norm["uns_hydrogens"]
            - df_norm["d_rmsd"]
            #- 0.5 * df_norm["binder_sasa"]
            #+ 0.6 * df_norm["delta_delta_interaction"]
            #- 0.5 * df_norm['helicity']
            + 10 * df_norm['ipsae']
            #+ df_norm["iptm_af"]
            + df_norm["lis"]
            #+ df_norm["d0res"]
            #+ df_norm["d0dom"]
            + df_norm["iptm_d0chn"]
            #+ df_norm["pdockq2"]
            #+ df_norm["pdockq"]
            #+ df_norm["ptm"]
            - df_norm["t_pae"]
            #+ 1 * df_norm["numbers_clusters"]
            #- df_norm["seq_identity"]
        ) / 9
    print(df_norm["score"])

    rows = []

    for idx, entry in df_norm.iterrows():
        sequence = entry["sequence"]
        reward = entry["score"]

        rows.append({
            "prompt": label,
            "completion": format_sequence(sequence, label),
            "reward":reward
        })

    return Dataset.from_list(rows)


seed_everything(CONFIG["seed"])


dataset = generate_dataset(args.iteration_num, args.label)
split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)

train_dataset = split['train']
eval_dataset   = split['test']

tokenizer_dir = args.model_dir
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                          add_eos_token=False, # NEED this for training NOT for generate() else add eos at the end of promt
                                          add_bos_token=False,
                                          use_fast=True)

# Determine model path
if args.iteration_num > 1 :
    model_path = f"{args.d_path}output_iteration{args.iteration_num-1}"
else:
    model_path = args.model_dir

print(f"Loading model from: {model_path}")


# Apply LoRA ONLY on the first iteration
if args.iteration_num == 1:
    
    print("Applying LoRA configuration for the first time.")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Targeting attention and MLP layers
    bias="none",
    task_type="CAUSAL_LM")
    
    # Wrap the base model with LoRA
    model = get_peft_model(model, lora_config)

else: 

    print("Loading model and injecting peft layers")
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, model_path, is_trainable=True)


lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)
current_lr = lr_list[args.iteration_num-1].item()

optimizer = AdamW(
    model.parameters(), # Pass the model's parameters to the optimizer
    lr=current_lr,
    betas=CONFIG["adam_betas"],
    eps=CONFIG["epsilon"],
    weight_decay=CONFIG["adam_decay"]
)

# A simple linear scheduler
num_training_steps = len(train_dataset) * CONFIG["num_epochs"]
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


training_args = GRPOConfig(output_dir=f"{args.d_path}output_iteration{args.iteration_num}",
                           logging_steps=100,
                           beta=CONFIG["beta"],
                           num_train_epochs = CONFIG["num_epochs"],
                           learning_rate = CONFIG["learning_rate"], # Use the calculated learning rate
                           do_train = True,
                           do_eval = True,
                           eval_strategy = "epoch",
                           save_strategy = "steps",
                           eval_steps = 500,
                           save_total_limit = 1,
                           save_steps = 5,
                           per_device_train_batch_size = 25,
                           per_device_eval_batch_size = 25,
                           num_generations = 5,
                           gradient_accumulation_steps=1
)

#print("model ",model)
trainer = pLM_GRPOTrainer(
    model= model,
    ref_model = args.model_dir,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer,
    optimizers = (optimizer, scheduler) # Pass the newly created optimizer and scheduler
    )

# The trainer will use the optimizers passed to it.
# No need to manually set them again.
# trainer.lr_scheduler       = scheduler
# trainer.lr_scheduler_state = None

trainer.train()

# This will correctly save the LoRA adapters and configuration.
trainer.save_model()

torch.cuda.empty_cache()