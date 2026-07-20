from typing import Any, Literal
import torch
import math
import torch.nn as nn
from dataclasses import dataclass, field
import logging
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

if is_peft_available():
    from peft import PeftConfig

logger = logging.getLogger(__name__)


@dataclass
class ProtRL_REINFORCETrainingArgument(ProtRLTrainingArgument):
    pass
    #entropy_bonus: float = field(
    #    default=0.0,
    #    metadata={
    #        "help": "Coefficient for entropy bonus. Positive values encourage exploration "
    #                "by penalising low-entropy (overconfident) distributions. Set to 0 to disable."
    #    },
    #)


class ProtRL_REINFORCETrainer(ProtRLBaseTrainer):
    """
    REINFORCE trainer for offline protein RL.

    Loss: L = -E[log pi(a|s) * A(s,a)]

    where A(s,a) = (r - mean(r)) / (std(r) + eps) is the standardised advantage.
    An optional entropy bonus can be added to encourage exploration.

    Unlike GRPO and wDPO, REINFORCE does not apply a KL penalty (beta is unused).
    A reference model is still created by the base class — it is used only to compute
    logging metrics (implicit reward correlation) but not in the loss itself.
    """

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        processing_class: PreTrainedTokenizerBase | None,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        args: ProtRL_REINFORCETrainingArgument | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | None = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple = (None, None),
        optimizer_cls_and_kwargs: tuple | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        peft_config: "PeftConfig | None" = None,
    ):
        if args is None:
            args = ProtRL_REINFORCETrainingArgument(output_dir="tmp_reinforce_trainer")

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

        self.entropy_bonus = args.entropy_bonus

    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, list | torch.LongTensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        REINFORCE loss with optional entropy bonus.

        Loss = -mean_over_batch(sum_over_tokens(log_pi * advantage))
             - entropy_bonus * H(pi)
        """
        metrics = {}

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        model_output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **model_kwargs,
        )
        policy_logits = model_output.logits

        # Per-token log probabilities for completion tokens (prompt masked out)
        per_token_logps = self.get_batch_logps(policy_logits, inputs["labels"])

        # extract reward
        rewards = inputs["reward"]

        # REINFORCE: sum log-probs over tokens, weight by advantage, average over batch
        seq_logps = per_token_logps.sum(dim=-1)
        loss = -(seq_logps * rewards).mean()

        ## Optional entropy bonus: H(pi) = -sum_v(pi * log pi) averaged over completion tokens
        #if self.entropy_bonus != 0.0:
        #    completion_mask = (inputs["labels"][..., 1:] != -100).float()
        #    n_completion_tokens = completion_mask.sum().clamp(min=1)
        #    log_probs = policy_logits[..., :-1, :].log_softmax(dim=-1)
        #    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        #    entropy = (entropy * completion_mask).sum() / n_completion_tokens
        #    loss = loss - self.entropy_bonus * entropy

        if self.aux_loss_enabled:
            loss = loss + self.router_aux_loss_coef * model_output.aux_loss

        # Logging: implicit reward correlation and log-p correlation
        ref_logits = self.compute_ref_logits(inputs)
        per_token_ref_logps = self.get_batch_logps(ref_logits, inputs["labels"])
        log_ratios = (per_token_logps - per_token_ref_logps).sum(dim=-1)

        all_log_ratios = self.accelerator.gather_for_metrics(log_ratios.detach())
        all_log_p = self.accelerator.gather_for_metrics(seq_logps.detach())
        all_rewards = self.accelerator.gather_for_metrics(rewards.detach())

        # Compute spearman correlation between i_rwd as well as log_p and true rwd
        i_rwd_corr_val = spearman_correlation(all_log_ratios, all_rewards).item()
        log_p_corr_val  = spearman_correlation(all_log_p, all_rewards).item()

        prefix = "eval_" if train_eval == "eval" else ""

        # skip NaN correlation to avoid polluting log
        if not math.isnan(i_rwd_corr_val):
            metrics[f"{prefix}i_reward_correlation"] = i_rwd_corr_val
        if not math.isnan(log_p_corr_val):
            metrics[f"{prefix}logp_correlation"] = log_p_corr_val

        metrics[f"{prefix}true_rwd_mean"] = all_rewards.mean().item()
        metrics[f"{prefix}true_rwd_std"] = all_rewards.std().item() if all_rewards.numel() > 1 else 0.0

        return loss, metrics
