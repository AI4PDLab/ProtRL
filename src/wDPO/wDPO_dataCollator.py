import torch
from typing import Any, Dict, List
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

@dataclass
class wDPODataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: int | None = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_input_ids = []
        all_labels = []
        all_rewards = []

        for feature in features:
            # 1. Tokenize prompt and completion separately
            # We assume features['prompt'] and features['completion'] are strings
            prompt_ids = self.tokenizer.encode(feature["prompt"], add_special_tokens=False)
            completion_ids = self.tokenizer.encode(feature["completion"], add_special_tokens=False)

            # 2. Concatenate
            # NOTE: Always add BOS/EOS 
            input_ids = [self.tokenizer.bos_token_id] + prompt_ids + completion_ids + [self.tokenizer.eos_token_id]

            # 3. Create Labels (masking the prompt with -100)
            # NOTE: 1+len(prompt_ids), added 1 to include BOS token
            labels = ([-100] * (1+len(prompt_ids))) + completion_ids + [self.tokenizer.eos_token_id]

            all_input_ids.append(torch.tensor(input_ids))
            all_labels.append(torch.tensor(labels))
            all_rewards.append(feature["reward"])

        # 4. Dynamic Padding
        batch = self.tokenizer.pad(
            {"input_ids": all_input_ids},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Pad labels manually using -100
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=-100
        )

        batch["reward"] = torch.tensor(all_rewards, dtype=torch.float32)

        return batch
