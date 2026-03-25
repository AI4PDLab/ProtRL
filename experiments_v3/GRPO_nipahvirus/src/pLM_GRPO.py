from typing import Any, Callable, Optional, Union
import torch
import torch.nn.functional as F

import types
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset, IterableDataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizerBase, TrainerCallback
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from torch.utils.data import Sampler, RandomSampler

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from trl.models import create_reference_model            

from transformers import AutoTokenizer
from trl.trainer.utils import pad
from torch import nn
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from typing import Any, Optional, Union
import numpy as np 

class pLM_GRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[str, PreTrainedModel, Callable[..., list[float]]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[Any] = None,
        *,
        ref_model: Union[str, PreTrainedModel],
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # Reference model
        model_init_kwargs = args.model_init_kwargs or {}
        #ref_model = AutoModelForCausalLM.from_pretrained(ref_model).to("cuda")

        if self.beta == 0.0:
            self.ref_model = None
        else:
            # accept either a name or an already-loaded model
            self.ref_model = (ref_model if isinstance(ref_model, PreTrainedModel)
                              else AutoModelForCausalLM.from_pretrained(ref_model, **model_init_kwargs))
            self.ref_model.to(self.accelerator.device)
            for p in self.ref_model.parameters():
                p.requires_grad = False
            self.ref_model.eval()

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:

        if dataset is None:
            dataset = self.train_dataset
            
        return RandomSampler(self.train_dataset)
    
    def _get_eval_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:

        if dataset is None:
            dataset = self.eval_dataset
            
        return RandomSampler(self.eval_dataset)


    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
        ) -> dict[str, Union[torch.Tensor, Any]]:
        
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"
        prompts = [x["prompt"] for x in inputs]

        prompt_inputs = self.processing_class(text=prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)
        
        completions = [x["completion"] for x in inputs]
        completions_input = self.processing_class(text=completions, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
        completions_ids, completions_mask = completions_input["input_ids"].to(device), completions_input["attention_mask"].to(device)

        #completions_ids = [ids.to(device) for ids in completions_ids]
        #completions_ids = pad(completions_ids, padding_value=self.processing_class.pad_token_id)
        
        prompt_completion_ids = torch.cat([prompt_ids, completions_ids], dim=1).to(device).int()
        attention_mask = torch.cat([prompt_mask, completions_mask], dim=1).to(device).int()
        
        rewards = torch.tensor([x["reward"] for x in inputs], device=device)  # shape [N*K]
        K = getattr(self.args, "num_generations_per_prompt", completions_ids.size(0))
        N = rewards.numel() // K
        rewards_grouped = rewards.view(N, K)                                   # [N, K]
        mean_grouped_rewards = rewards_grouped.mean(dim=1)                     # [N]
        std_grouped_rewards  = rewards_grouped.std(dim=1, unbiased=False)      # [N]
        advantages = ((rewards_grouped - mean_grouped_rewards.unsqueeze(1)) /
                    (std_grouped_rewards.unsqueeze(1) + 1e-4)).reshape(-1)   

        is_eos = completions_ids == self.processing_class.eos_token_id
        logits_to_keep = completions_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None


            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completions_mask = self.accelerator.gather_for_metrics(completions_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completions_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completions_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completions_mask.float().max().item())

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completions_mask = agg_completions_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completions_mask) / len(agg_completions_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completions_mask) == 0:
            # edge case where no completed sequences are found
            term_completions_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completions_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completions_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completions_mask.float().max().item())
        
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts))
        self._textual_logs["completion"].extend(gather_object(completions))
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completions_ids,
            "completion_mask": completions_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def _embedding_diversity_loss(self, model, input_ids, attention_mask=None, alpha_div=0.9999):
                if alpha_div == 0.0:
                        # return a scalar zero on the right device/dtype
                        return input_ids.new_zeros(())
                out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                H = out.hidden_states[-1]                       
                m = attention_mask.unsqueeze(-1).to(H.dtype)   
                emb = (H * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  
                if emb.size(0) < 2:
                    return emb.new_zeros(())
                z = F.normalize(emb, p=2, dim=1)               
                cos = z @ z.T                                   
                triu = torch.triu(torch.ones_like(cos, dtype=torch.bool), diagonal=1)
                mean_cos = cos[triu].mean()                     
                diversity = 1.0 - mean_cos                      
            
                return -alpha_div * diversity      

    def _compute_loss(self, model, inputs):
                    
            # Compute the per-token log probabilities for the model
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

            # Compute the KL divergence between the model and the reference model
            if self.beta != 0.0:
                ref_per_token_logps = inputs["ref_per_token_logps"]
                per_token_kl = (ref_per_token_logps - per_token_logps)

            # Compute the loss
            advantages = inputs["advantages"]
            alpha_div = getattr(self.args, "alpha_div", 0.9999)
            emb_loss = self._embedding_diversity_loss(model, input_ids, attention_mask, alpha_div)

            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
            # _generate_and_score_completions) and use per_token_logps.detach() instead.
            old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
            
            # ratios and advantage broadcast (match per-token shape)
            B, T = per_token_logps.shape
            ratio = torch.exp(per_token_logps - old_per_token_logps)          # [B, T]
            ratio_clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
            adv = advantages.unsqueeze(1).expand(B, T)                        # [B, T]
            per_token_loss1 = -ratio * adv
            per_token_loss2 = -ratio_clipped * adv
            per_token_loss = torch.minimum(per_token_loss1, per_token_loss2)  # [B, T]
            
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl

            if self.loss_type == "grpo":
                loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            elif self.loss_type == "bnpo":
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            elif self.loss_type == "dr_grpo":
                loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            # Log the metrics
            mode = "eval" if self.control.should_evaluate else "train"

            if self.beta != 0.0:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

            # Compute the clipped probability ratios
            is_low_clipped  = (ratio < 1 - self.epsilon_low) & (adv < 0)
            is_high_clipped = (ratio > 1 + self.epsilon_high) & (adv > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            den = completion_mask.sum()
            low_clip   = (is_low_clipped  & completion_mask.bool()).sum() / den
            high_clip  = (is_high_clipped & completion_mask.bool()).sum() / den
            clip_ratio = (is_region_clipped & completion_mask.bool()).sum() / den

            gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            #self._metrics[mode]["clip_ratio/low_min"].append(np.nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            #self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
            return loss + 10*emb_loss