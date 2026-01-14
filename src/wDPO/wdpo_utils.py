import logging
import torch
import torch.nn as nn
import transformers
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from copy import deepcopy
from transformers import (
    AutoConfig,
    PreTrainedModel,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
import torch.nn.functional as F
from datasets import Dataset


def create_wDPO_dataset(df, prompt):
    """
    Based on ProtRL wDPO Trainer required structure
    """
    rows = []
    for idx, entry in df.iterrows():
        sequence = entry["sequence"]
        activity = entry["activity"]

        rows.append({
            "prompt": prompt,
            "completion": sequence,
            "reward": activity
        })

    return Dataset.from_list(rows)

def compute_wDPO_metrics(eval_preds):
    """
    Custom evaluation metric for wDPO computing the spearman correlation between intristic rewards and activity
    
    :param eval_preds:  contains all log_ratios from the whole eval set and 
    """

    # logits here is actually our log_ratios
    # labels here is actually our rewards
    # based on wDPO.predition_step
    logits, labels = eval_preds
    
    # Convert to torch for our helper function
    # device is managed automatically by the Hugging Face Trainer's evaluation loop.
    log_ratios = torch.tensor(logits)
    rewards = torch.tensor(labels)
    
    correlation = spearman_correlation(log_ratios, rewards)
    
    return {
        "reward_correlation": correlation.item()
    }


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Computes Spearman Rank Correlation in PyTorch."""
    def get_ranks(v: torch.Tensor):
        # We use argsort twice to get the ordinal ranking
        # the first returns the indices that would sort the tensor
        # the second, given that order, gives you the elemt's position (i.e., the rank) 
        return v.argsort(dim=-1).argsort(dim=-1).float()

    # Extra the ranks
    x_rank = get_ranks(x)
    y_rank = get_ranks(y)
    
    # Pearson correlation on ranks = Spearman colleration
    combined = torch.stack([x_rank, y_rank])
    corr = torch.corrcoef(combined)[0, 1]
    return corr

def create_model_from_path(
    model_id: str, architecture: _BaseAutoModelClass | None = None, **kwargs
) -> PreTrainedModel:
    """
    Create a model from a given path using the specified initialization arguments.

    Args:
        model_id (`str`):
            Path to the model. Can be either a local directory or a model identifier from the Hugging Face Hub.
        architecture (`_BaseAutoModelClass` or `None`, *optional*):
            Model architecture class to instantiate. The model is initialized using the `from_pretrained` method of
            this class. If `None`, the architecture will be inferred from the model's configuration.
        kwargs (`dict`):
            Initialization keyword arguments to pass to the model's `from_pretrained` method. When `'dtype'` is
            specified, it can be either a `torch.dtype` or one of the strings: `'bfloat16'`, `'float16'`, `'float32'`,
            or `'auto'`.

    Returns:
        [`~transformers.PreTrainedModel`]:
            The instantiated model.
    """
    dtype = kwargs.get("dtype", "auto")
    if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
        pass  # dtype is already a torch.dtype or "auto" or None
    elif isinstance(dtype, str) and dtype in ["bfloat16", "float16", "float32"]:
        kwargs["dtype"] = getattr(torch, dtype)
    else:
        raise ValueError(
            "Invalid `dtype` passed to the config. Expected either 'auto' or a string representing "
            f"a valid `torch.dtype` (e.g., 'float32'), but got {dtype}."
        )
    kwargs["device_map"] = kwargs.get("device_map", "auto")
    if architecture is None:
        config = AutoConfig.from_pretrained(model_id)
        architecture = getattr(transformers, config.architectures[0])
    model = architecture.from_pretrained(model_id, **kwargs)
    return model

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


LAYER_PATTERNS = [
    "transformer.h.{layer}",
    "model.decoder.layers.{layer}",
    "gpt_neox.layers.{layer}",
    "model.layers.{layer}",
]


def create_reference_model(
    model: nn.Module, num_shared_layers: int | None = None, pattern: str | None = None
) -> nn.Module:
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    Args:
        model ([`nn.Module`]): The model to be copied.
        num_shared_layers (`int`, *optional*):
            The number of initial layers that are shared between both models and kept frozen.
        pattern (`str`, *optional*): The shared layers are selected with a string pattern
            (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    Returns:
        [`nn.Module`]
    """
    if is_deepspeed_zero3_enabled():
        raise ValueError(
            "DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoModelForCausalLM.from_pretrained()`."
        )

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    # if no layers are shared, return copy of model
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()

    # identify layer name pattern
    if pattern is not None:
        pattern = pattern.format(layer=num_shared_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
            if any(pattern_candidate in name for name in parameter_names):
                pattern = pattern_candidate
                break

    if pattern is None:
        raise ValueError("Layer pattern could not be matched.")

    # divide parameters in shared and unshared parameter lists
    shared_param_list = []
    unshared_param_list = []

    shared_parameter = True
    for name, _param in model.named_parameters():
        if pattern in name:
            shared_parameter = False
        if shared_parameter:
            shared_param_list.append(name)
        else:
            unshared_param_list.append(name)

    # create reference of the original parameter if they are shared
    for param_name in shared_param_list:
        param = model.get_parameter(param_name)
        param.requires_grad = False

        _ref_param = ref_model.get_parameter(param_name)

    # for all other parameters just make sure they don't use gradients
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False

    if pattern is not None and len(unshared_param_list) == 0:
        logging.warning("Pattern passed or found, but no layers matched in the model. Check for a typo.")

    return ref_model.eval()