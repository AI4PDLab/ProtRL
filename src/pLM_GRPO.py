from typing import Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
import logging
from contextlib import contextmanager
from src.ProtRL_Trainer import ProtRLTrainingArgument, ProtRLBaseTrainer, spearman_correlation
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    DataCollator,
)
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available
from datasets import Dataset, IterableDataset
from collections.abc import Callable 
import math

if is_peft_available():
    from peft import (
        PeftConfig,
    )

logger = logging.getLogger(__name__)

"""
#TODO
1. Check batch sampler is working correctly (ideally check this on distributed training) - may want to extend this to all methods
2. Implement GRPO loss, since n. iterations=1, don't need KL relative to old model, so simpler implementation ( may want to change it in the future)

Disclaimer: GRPO is not an off-policy method (e.g., the REINFORCE-like gradient update is on-policy), after the first batch update, any other sample becomes off-policy, but this seems  to still work fine
"""
@dataclass
class ProtRL_GRPOTrainingArgument(ProtRLTrainingArgument):
    """
    Inherit to rename
    """
    pass


class ProtRL_GRPOTrainer(ProtRLBaseTrainer):
    """
    Class to implement an "offline" version of GRPO (i.e., perfom GRPO update on fixed prompts)
    """
    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        processing_class: PreTrainedTokenizerBase | None,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        args: ProtRLTrainingArgument | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
    ):

        if args is None:
            output_dir = "tmp_grpo_trainer"
            logger.info(f"No `ProtRLTrainingArgument` passed, using `output_dir={output_dir}`.")
            args = ProtRLTrainingArgument(output_dir=output_dir)

        # Initialize parent class
        super().__init__(
            model=model,
            processing_class=processing_class,
            ref_model=ref_model,
            args=args,  
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )


    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, list | torch.LongTensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Implement loss for (offline) GRPO
        """

        metrics = {}

        # turn off cashing of key-value, only useful for 
        # for auto-regressive generation & waste memory
        # for offline RL method (i.e., no generation)
        model_kwargs = {"use_cache": False}

        # Need this for MoE
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"]= True

        # 1. Get logits model
        model_output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **model_kwargs,
        )

        policy_logits = model_output.logits

        # 2. get logits for reference model
        ref_logits = self.compute_ref_logits(inputs)

        # 3. Calculate Log Probabilities for the completion part
        per_token_policy_logps = self.get_batch_logps(policy_logits, inputs["labels"])
        per_token_ref_logps = self.get_batch_logps(ref_logits, inputs["labels"])

        # 4. Compute KL between model and reference
        # not for PAD and prompt log_p = 0 in get_batch_logps so KL=0 for those tokens
        if self.beta != 0.0:
            per_token_kl = torch.exp(per_token_ref_logps - per_token_policy_logps) - (per_token_ref_logps - per_token_policy_logps) - 1 

            # Explicitly mask out per-token KL for prompt and pad tokens using labels
            # although these already have zero KL divergence in the computation above
            # better to be explicit 
            per_token_kl = per_token_kl * (inputs["labels"][..., 1:] != -100)
        else:
            per_token_kl = 0

        # 5. COmpute advatange
        rewards = inputs["reward"]
        # safe-guard
        group_size = rewards.shape[-1]
        if group_size > 1:
            std = rewards.std(dim=-1)
        else:
            std = torch.ones_like(rewards)  # or zero out the advantage entirely
        mean = rewards.mean(dim=-1)
        advantage = (rewards - mean) / (std + 1e-12)

        # 6. Compute the GRPO Loss
        loss = (- 1 * (per_token_policy_logps * advantage.unsqueeze(-1)) + self.beta * per_token_kl).mean() 

        if self.aux_loss_enabled:
            loss = loss + self.router_aux_loss_coef * model_output.aux_loss

        ## ---- COmpute useful logging statistic ----
        # compute log ratio for correlation metrics
        log_ratios = self.beta * (per_token_policy_logps - per_token_ref_logps)

        # Gather across all processes for a more stable correlation
        # This ensures we are correlating the full macro-batch
        all_log_ratio = self.accelerator.gather_for_metrics(log_ratios.sum(dim=-1)).detach()
        all_log_p = self.accelerator.gather_for_metrics(per_token_policy_logps.sum(dim=-1)).detach()
        all_rewards = self.accelerator.gather_for_metrics(rewards).detach()

        # Compute spearman correlation between i_rwd as well as log_p and true rwd
        i_rwd_corr_val = spearman_correlation(all_log_ratio, all_rewards).item()
        log_p_corr_val  = spearman_correlation(all_log_p, all_rewards).item()

        prefix = "eval_" if train_eval == "eval" else ""

        # skip NaN correlation to avoid polluting log
        if not math.isnan(i_rwd_corr_val):
            metrics[f"{prefix}i_reward_correlation"] = i_rwd_corr_val
        if not math.isnan(log_p_corr_val):
            metrics[f"{prefix}logp_correlation"] = log_p_corr_val

        metrics[f"{prefix}log"] =  all_log_p.mean().item()
        metrics[f"{prefix}true_rwd_mean"] =  all_rewards.mean().item()
        metrics[f"{prefix}true_rwd_std"] = all_rewards.std().item() if all_rewards.numel() > 1 else 0.0

        return loss, metrics
