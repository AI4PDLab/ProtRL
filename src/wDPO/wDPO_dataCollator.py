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
    """
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: int | None = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract rewards; they don't get padded
        rewards = [f.pop("reward") for f in features]

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

        
        batch["reward"] = torch.tensor(rewards, dtype=torch.float32)
        return batch
