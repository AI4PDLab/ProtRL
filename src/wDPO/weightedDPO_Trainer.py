from typing import Any, Literal 
import torch
import torch.nn as nn
from torch import autocast
import torch.nn.functional as F
from contextlib import contextmanager, nullcontext
import logging
from dataclasses import dataclass, field
from collections.abc import Callable
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.trainer_utils import EvalLoopOutput

from src.wDPO.wdpo_utils import peft_module_casting_to_bf16,  create_reference_model, disable_dropout_in_model,create_model_from_path, spearman_correlation, compute_wDPO_metrics 
from src.wDPO.wDPO_dataCollator import wDPODataCollatorWithPadding
from transformers.utils import is_peft_available
import inspect


from datasets import Dataset, IterableDataset
if is_peft_available():
    from peft import (
        PeftConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
import warnings

logger = logging.getLogger(__name__)

@dataclass
class wDPOTrainingArgument(TrainingArguments):
    beta: float = field(
            default=0.1,
            metadata={
                "help": "Parameter controlling the deviation from the reference model. "
                        "Higher β means less deviation from the reference model."
            },
        )
    # Parameters that control the model and reference model
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `wDPOTrainer` is provided as a string."
        },
    )
    ref_model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `ref_model` argument "
            "of the `wDPOTrainer` is provided as a string."
        },
    )
    model_adapter_name: str | None = field(
        default=None,
        metadata={"help": "Name of the train target PEFT adapter, when using LoRA with multiple adapters."},
    )
    ref_adapter_name: str | None = field(
        default=None,
        metadata={"help": "Name of the reference PEFT adapter, when using LoRA with multiple adapters."},
    )

    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )

    # NOTE: If you set this value, greater_is_better will default to True unless the name ends with “loss”. Don’t forget to set it to False if your metric is better when lower.
    metric_for_best_model: str | None = field(
        default="reward_correlation", metadata={"help": "The metric to use to compare two different models."}
    )

    # Default to True since use Spearman correlation to pick best model (this should be by default when passing metric_for_best_model that doesn't start with loss - but just in case)
    greater_is_better: bool | None = field(
        default=True, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )



class wDPOTrainer(Trainer):
    """
    Override Trainer class to implement an emergency checkpoint that is trigger when the job is about to end by setting self.control.should_save to True when the time comes. Override both `train_step()` and `prediction_step` to check if job is about to end.
    """
    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        processing_class: PreTrainedTokenizerBase,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        args: wDPOTrainingArgument | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
    ):

        if compute_metrics is not None:
            raise ValueError("wDPO uses a custom compute metrics function by default, please don't pass any compute_metrics")

        compute_metrics = compute_wDPO_metrics 

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = wDPOTrainingArgument(output_dir=output_dir)

        # this is key to pass the necessary columns
        args.remove_unused_columns = False

        if not args.greater_is_better and args.load_best_model_at_end:
            warnings.warn("WARNING: greater_is_better=False, since wDPO uses custom spearman correlation metric to pick the best model, this will result in loading the wost model at the end", UserWarning)
        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Special case for DeepSpeed: requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type == "DEEPSPEED":
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Reference model
        if isinstance(ref_model, str):
            model_init_kwargs = args.ref_model_init_kwargs or {}
            # Special case for DeepSpeed: requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type == "DEEPSPEED":
                model_init_kwargs["device_map"] = None
            ref_model = create_model_from_path(ref_model, **model_init_kwargs)
        else:
            if args.ref_model_init_kwargs is not None:
                logger.warning(
                    "You passed `ref_model_init_kwargs` to the `DPOConfig`, but your model is already instantiated. "
                    "The `ref_model_init_kwargs` will be ignored."
                )
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you can simply omit the `ref_model` argument and it will be created for you."
            )

        # PEFT configuration and model wrapping
        model = self._prepare_peft_model(model, ref_model, peft_config, args) 

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)


        # Disable dropout in the model and reference model
        if args.disable_dropout:
            if self.ref_model is not None:
                disable_dropout_in_model(model)
                disable_dropout_in_model(self.ref_model)

        if args.use_liger_kernel:
            raise ValueError("Liger kernel currently not implmented for wDPO loss")

        self.beta = args.beta

        # Data collator
        data_collator = wDPODataCollatorWithPadding(processing_class)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )


    def _prepare_peft_model(
        self, model: PreTrainedModel, ref_model: PreTrainedModel, peft_config: Any, args: wDPOTrainingArgument
    ) -> PreTrainedModel:
        """Prepares a model for PEFT training."""
        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            else:
                model = self._prepare_gradient_checkpointing(model, args)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        else:
            model = self._prepare_gradient_checkpointing(model, args)

        return model

    def _prepare_gradient_checkpointing(self, model: PreTrainedModel, args: wDPOTrainingArgument):
        """Prepare the gradienting checkpointing for the model."""
        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        if args.gradient_checkpointing:
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Computes the log probabilities of the gold tokens (completion part).
        labels: shape (batch, seq_len) where prompt is -100
        """
        # Shift logits and labels so that tokens predict the next token
        # Logits: [B, L, V] -> [B, L-1, V]
        # Labels: [B, L]   -> [B, L-1] (excluding the first token which has no target)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Loss mask: False for any -100 label (i.e., mask out prompt/padding)
        loss_mask = (shift_labels != -100) 

        # Standardize labels to 0 for gathering (we will mask them out anyway)
        dummy_labels = shift_labels.clone()
        dummy_labels[dummy_labels == -100] = 0

        # compute log probabilities
        all_log_probs = shift_logits.log_softmax(-1)

        # use label to retrieve probs of generated tokens
        gen_logps = torch.gather(
            all_log_probs, dim=2, index=dummy_labels.unsqueeze(2)
        ).squeeze(2)

        # Sum log probabilities for the completion only
        return (gen_logps * loss_mask).sum(-1)
    

    def compute_ref_logits(self, model, batch: dict[str, torch.LongTensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), compte_ref_context_manager:
            # Use Lora model
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    ).logits
            else:
                # use Ref model
                ref_logits = self.ref_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).logits

        return ref_logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled:
            raise ValueError("Sequence parallelism is currently not supported for wDPO")

        # 1. Get logits from both models
        policy_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        ).logits

        ref_logits = self.compute_ref_logits(model,inputs)

        # 2. Calculate Log Probabilities for the completion part
        policy_logps = self.get_batch_logps(policy_logits, inputs["labels"])
        ref_logps = self.get_batch_logps(ref_logits, inputs["labels"])

        # 3. Compute the wDPO Loss
        # log_ratios: (pi_policy / pi_ref)
        log_ratios = self.beta * (policy_logps)# - ref_logps)

        # Weighted DPO Loss
        rewards = inputs["reward"].detach() # just in case

        # softmax the rewards to get a distribution
        weights = torch.softmax(rewards, dim=0)
        loss = -1 * F.cross_entropy(log_ratios, weights) # multiple by -1 since want to max

        ## ---- COmpute useful logging statistic ----
        # Gather across all processes for a more stable correlation
        # This ensures we are correlating the full macro-batch
        _log_rations = log_ratios.detach()
        all_log_ratios = self.accelerator.gather(_log_rations)
        all_rewards = self.accelerator.gather(rewards)

        if self.state.global_step % self.args.logging_steps == 0:
            corr = spearman_correlation(all_log_ratios, all_rewards)
            # Log to W&B / TensorBoard
            self.log({"train/reward_correlation": corr.item()})

        return (loss, {"logits": policy_logits}) if return_outputs else loss


    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:

        if ignore_keys is not None:
            raise ValueError("wDPO currently does not support keys_to_ignore_at_inference")

        prediction_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        with torch.no_grad(), prediction_context_manager:
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            policy_logits = outputs["logits"]
            ref_logits = self.compute_ref_logits(model,inputs)

            policy_logps = self.get_batch_logps(policy_logits, inputs["labels"])
            ref_logps = self.get_batch_logps(ref_logits, inputs["labels"])
            log_ratios = (policy_logps - ref_logps)
            rewards = inputs["reward"]

        if prediction_loss_only:
                return (loss.detach(), None, None)

        # Return log_ratios as 'logits' and rewards as 'labels' 
        # The Trainer will gather these from ALL steps and ALL GPUs
        return (loss.detach(), log_ratios, rewards) 



