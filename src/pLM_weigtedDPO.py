from typing import Any, Callable, Optional, Union
import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from torch.utils.data import Sampler, RandomSampler
from trl import GRPOConfig, GRPOTrainer
from trl.models import create_reference_model            
from trl.trainer.utils import pad
from accelerate.utils import gather_object

class weighted_DPO(GRPOTrainer):
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

        # Reference model setup
        model_init_kwargs = args.model_init_kwargs or {}
        
        # Note: ref_model argument is shadowed here by the loaded model object
        # This assumes ref_model passed as argument is a string path
        ref_model_path = ref_model
        print(f"Loading reference model from {ref_model_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path).to("cuda")

        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            # If using DeepSpeed Zero3, load using the path and init kwargs
            self.ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, **model_init_kwargs)
        else:
            # Otherwise create reference model from the loaded model object
            self.ref_model = create_reference_model(ref_model)
    
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
        
        # Process prompts
        prompts = [x["prompt"] for x in inputs]
        prompt_inputs = self.processing_class(text=prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)
        
        # Process completions
        completions = [x["completion"] for x in inputs]
        completions_input = self.processing_class(text=completions, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
        completions_ids, completions_mask = completions_input["input_ids"].to(device), completions_input["attention_mask"].to(device)
        completions_ids = pad(completions_ids, padding_value=self.processing_class.pad_token_id)
        
        # Concatenate prompt and completion
        prompt_completion_ids = torch.cat([prompt_ids, completions_ids], dim=1).to(device)
        attention_mask = torch.cat([prompt_mask, completions_mask], dim=1).to(device)
        
        # Process rewards
        rewards = torch.tensor([x["reward"] for x in inputs], device=device)
                
        # Calculate grouped rewards for logging
        batch_size = rewards.shape[0] // completions_ids.shape[0] # This evaluates to 1
        rewards_grouped = rewards.view(batch_size, completions_ids.shape[0])
        mean_grouped_rewards = rewards_grouped.mean(dim=1)
        std_grouped_rewards  = rewards_grouped.std(dim=1, unbiased=False)

        # Prepare for logit computation
        is_eos = completions_ids == self.processing_class.eos_token_id
        logits_to_keep = completions_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )

        # Logging
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths
        agg_completions_mask = self.accelerator.gather_for_metrics(completions_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completions_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completions_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completions_mask.float().max().item())

        # Log terminated sequences
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completions_mask = agg_completions_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completions_mask) / len(agg_completions_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completions_mask) == 0:
            term_completions_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completions_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completions_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completions_mask.float().max().item())
        
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts))
        self._logs["completion"].extend(gather_object(completions))
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completions_ids,
            "completion_mask": completions_mask,
            "rewards": rewards,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "num_items_in_batch": torch.tensor(batch_size, device=device),
        }
    
    def _compute_loss(self, model, inputs, return_outputs=False):
        
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        batch_size = prompt_ids.shape[0]

        per_token_logps, _ = self._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, batch_size)
        ref_per_token_logps = inputs["ref_per_token_logps"]

        rewards = inputs["rewards"]

        if torch.allclose(ref_per_token_logps, per_token_logps, rtol=1e-5, atol=1e-8): 
            # print("equal") # Commented out to reduce noise
            pi_ratio = self.beta * per_token_logps.mean(1)
        else:
            pi_ratio = self.beta * (per_token_logps.mean(1) - ref_per_token_logps.mean(1))
        
        weights = torch.softmax(rewards, dim=0)
        loss = F.cross_entropy(pi_ratio, weights)

        if return_outputs:
            return loss, {"rewards": rewards}
        return loss
