import torch
from typing import Any, Dict, List
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

@dataclass
class wDPODataCollatorWithPadding:
    """
    Data collator for wDPO to return rewards as well as mask the prompt in the lables (i.e., only need to compute logs for completion)
    """
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: int | None = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_input_ids = []
        all_labels = []
        all_rewards = []

        # NOTE: Currently tokenize data here on the fly, later may wanna move this loop to a prepare_dataset()
        # function that tokenize data and create correct labels at start of training and, here, in data collator
        # only apply padding
        for feature in features:
            # 1. Tokenize prompt and completion separately
            # We assume features['prompt'] and features['completion'] are strings
            #prompt_ids = self.tokenizer.encode(feature["prompt"], add_special_tokens=False)
            #completion_ids = self.tokenizer.encode(feature["completion"], add_special_tokens=False)
            prompt_ids = self.tokenizer(feature["prompt"], add_special_tokens=False)['input_ids']
            completion_ids = self.tokenizer(feature["completion"], add_special_tokens=False)['input_ids']

            # 2. Concatenate
            # NOTE: Chekc whether need to add BOS and always add EOS
            if prompt_ids[0] == self.tokenizer.bos_token_id:
                input_ids =  prompt_ids + completion_ids + [self.tokenizer.eos_token_id]
                # 3. Create Labels (masking the prompt with -100)
                labels = ([-100] * (len(prompt_ids))) + completion_ids + [self.tokenizer.eos_token_id]
            else:
                input_ids = [self.tokenizer.bos_token_id] + prompt_ids + completion_ids + [self.tokenizer.eos_token_id]
                # 3. Create Labels (masking the prompt with -100)
                # NOTE: 1+len(prompt_ids), added len of BOS token id if this has been added to the prompt
                bos_len =  1 if isinstance(self.tokenizer.bos_token_id, int) else len(self.tokenizer.bos_token_id) 
                labels = ([-100] * (bos_len+len(prompt_ids))) + completion_ids + [self.tokenizer.eos_token_id]


            all_input_ids.append(torch.tensor(input_ids))
            all_labels.append(torch.tensor(labels))
            all_rewards.append(feature["reward"])

        # 4. Dynamic Padding
        batch = self.tokenizer.pad(
            {"input_ids": all_input_ids},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
            padding_side="right",
        )

        # Pad labels manually using -100
        # i.e., pad token labels should be -100
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=-100
        )

        batch["reward"] = torch.tensor(all_rewards, dtype=torch.float32)

        return batch
