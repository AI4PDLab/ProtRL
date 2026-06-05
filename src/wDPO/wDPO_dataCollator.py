import torch
from typing import Any, Dict, List
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded 

@dataclass
class wDPODataCollatorWithPadding:
    """
    Data collator for wDPO to return rewards as well as mask the prompt in the lables (i.e., only need to compute logs for completion)
    This has been modified to return any extra field that is passed in the dataset (e.g., backbone etc.) allowing great flexibility
    """
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: int | None = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. Extract rewards; they don't get padded
        rewards = [f.pop("reward") for f in features]

        # 2. Separate labels from features to pad them manually
        labels = [f.pop("labels") for f in features]

        # 3. Extract input_ids (needs padding) 
        padding_features = []
        extra: Dict[str, list] = {}

        for f in features:
            padding_features.append({"input_ids": f.pop("input_ids")})
            for key, val in f.items():
                # store any extra field (e.g., backbone etc.) to be returned without padding
                extra.setdefault(key, []).append(val)

        # 4. only pad extracted input_ids
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padding_features,
            padding=self.padding,
            return_tensors="pt", # pytorch compatibility only
        )

        # 5. process labels and reward
        labels = [torch.tensor(l) for l in labels]
        # set padding labels directly to -100
        batch['labels']= torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        batch["reward"] = torch.tensor(rewards, dtype=torch.float32)

        # 6. Pass through any unknown extra fields as-is.
        #    Attempt to stack into a tensor; fall back to a plain list if that fails
        #    (e.g. strings, ragged arrays, nested dicts)
        for key, values in extra.items():
            try:
                batch[key] = torch.tensor(values)
            except (ValueError, TypeError):
                batch[key] = values

        return batch

        
@dataclass
class uncertainty_wDPODataCollatorWithPadding:
    """
    Data collator for sampling rewards based on a reward distribution, to be used with wDPOTrainer when uncertainty_aware=True
    """
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: int | None = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract rewards; they don't get padded
        rwd_statistics = [f.pop("reward_statistics") for f in features]

        # convert to tensors
        rwd_means = torch.tensor([m for m, s in rwd_statistics], dtype=torch.float32)
        rwd_stds  = torch.tensor([s for m, s in rwd_statistics], dtype=torch.float32)

        # sample
        reward = rwd_means + rwd_stds * torch.randn_like(rwd_stds)        

        # 2. Separate labels from features to pad them manually
        labels = [f.pop("labels") for f in features]

        # This single call pads input_ids, labels, AND create attention_mask
        #batch = self.tokenizer.pad(
        #    features,
        #    padding=self.padding,
        #    return_tensors="pt",
        #)
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            return_tensors="pt", # pytorch compatibility only
        )

        labels = [torch.tensor(l) for l in labels]
        # set padding labels directly to -100
        batch['labels']= torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        ## use tokenizer attention mask to mask padding token labels to ensure
        # only PAD tokens get masked
        #if "labels" in batch:
        #    batch["labels"][batch["attention_mask"] == 0] = -100

        batch["reward"] = reward
        
        return batch
