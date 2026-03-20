from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel)
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import torch
import numpy as np
import random
import pandas as pd
import math
import os
from pathlib import Path


def checkpoint_load(model_path):
   
    base_path = Path(model_path)
    for p in base_path.rglob("checkpoint-*"):
            if p.is_dir():
                checkpoint = p.resolve()
    
    return checkpoint



def load_optimizer_scheduler(model, checkpoint, lr, CONFIG):
    
    model = AutoModelForCausalLM.from_pretrained(model)

    optimizer = AdamW(
        model.parameters(),
        lr = lr,
        betas = CONFIG["adam_betas"],
        eps = CONFIG["epsilon"],
        weight_decay = CONFIG["adam_decay"]
    )

    if checkpoint is not None:
        optim_state_path = checkpoint / "optimizer.pt"
        saved_optim_state = torch.load(optim_state_path, map_location="cpu")
        optimizer.load_state_dict(saved_optim_state)
        for group in optimizer.param_groups:
                group["lr"] = lr
                group["initial_lr"] = lr

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    return optimizer, model, scheduler