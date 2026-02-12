from typing import Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import logging
from contextlib import contextmanager
from src.ProtRL_BaseTrainer import ProtRLTrainingArgument, ProtRLBaseTrainer 
from src.ProtRL_utils import spearman_correlation
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
    from peft import (
        PeftConfig,
    )

logger = logging.getLogger(__name__)

@dataclass
class wDPOTrainingArgument(ProtRLTrainingArgument):
    IRPO_regularisation: bool = field(
        default=False,
        metadata={"help": "Enable IRPO regularisation of likelihood of positive examples (adapted to wDPO)"}
    )

    IRPO_regulariser_coeff: float = field(
            default=0.05,
            metadata={
                "help": "Parameter controlling \alpha weight to IRPO regulariser"
            },
        )


class wDPOTrainer(ProtRLBaseTrainer):
    """
    Class to implement wDPO algorithm
    """
    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        processing_class: PreTrainedTokenizerBase | None,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        args: wDPOTrainingArgument | None = None,  # Type hint shows this is wDPOTrainingArgument
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
            output_dir = "tmp_wdpo_trainer"
            logger.info(f"No `wDPOTrainingArgument` passed, using `output_dir={output_dir}`.")
            args = wDPOTrainingArgument(output_dir=output_dir)

        # Initialize parent class
        super().__init__(
            model=model,
            processing_class=processing_class,
            ref_model=ref_model,
            args=args,  # Pass the wDPOTrainingArgument object
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

        # add  wDPOTrainingArgument specific argument
        self.IRPO_regularisation = self.args.IRPO_regularisation
        self.IRPO_reg_coeff = self.args.IRPO_regulariser_coeff


    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, list | torch.LongTensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Implement loss for wDPO
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
        policy_logps = self.get_batch_logps(policy_logits, inputs["labels"])
        ref_logps = self.get_batch_logps(ref_logits, inputs["labels"])


        # 4. Compute the wDPO Loss
        # log_ratios: (pi_policy / pi_ref)
        log_ratios = self.beta * (policy_logps - ref_logps)

        # Weighted DPO Loss
        rewards = inputs["reward"].detach() # just in case

        # softmax the rewards to get a distribution
        weights = torch.softmax(rewards, dim=0)
        loss = F.cross_entropy(log_ratios, weights) 

        if self.aux_loss_enabled:
            loss = loss + self.router_aux_loss_coef * model_output.aux_loss

        if self.IRPO_regularisation:
            # count n. of completion tokens using labels (i.e., prompt/pad labels set to -100)
            completion_mask = (inputs["labels"] != -100)
            completion_lengths = completion_mask.sum(dim=1).to(policy_logps.dtype).clamp(min=1) # use clamp to avoid division by 0
            normalized_policy_logps = policy_logps / completion_lengths
            # expectation over regulatisation E_w[L^{NN}]
            loss = loss - self.IRPO_reg_coeff * torch.sum(weights * normalized_policy_logps)


        ## ---- COmpute useful logging statistic ----
        # Gather across all processes for a more stable correlation
        # This ensures we are correlating the full macro-batch
        all_log_ratios = self.accelerator.gather_for_metrics(log_ratios).detach()
        all_rewards = self.accelerator.gather_for_metrics(rewards).detach()

        corr = spearman_correlation(all_log_ratios, all_rewards)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}reward_correlation"] =  corr.item()
        metrics[f"{prefix}log_ratio"] =  all_log_ratios.mean().item()
        metrics[f"{prefix}true_rwd_mean"] =  all_rewards.mean().item()
        metrics[f"{prefix}true_rwd_std"] =  all_rewards.std().item()
        #metrics[f"{prefix}log_ratio"] =  self.accelerator.gather_for_metrics(_log_rations).mean().item()
        #metrics[f"{prefix}true_rwd"] =  self.accelerator.gather_for_metrics(all_rewards).mean().item()

        return loss, metrics
