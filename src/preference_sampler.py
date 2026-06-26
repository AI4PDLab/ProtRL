import random
from torch.utils.data import Sampler
from collections import defaultdict
from datasets import Dataset, IterableDataset


class PreferenceBatchSampler(Sampler):
    """
    Sampler to organise batches with the same prompt (i.e., for preference comparison)
    """
    def __init__(self, dataset: Dataset | IterableDataset, batch_size: int, preference_col_name: str = "prompt", shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        """
        Args
            dataset: Dataset | IterableDataset containing the prompt + completion + reward sequences
            preference_col_name: name of the column in dataset used to build the preferences - entries with the same value/ID in this specified col will be treated as a preference set (i.e., compared against each other).
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Read only the column — don't iterate/decode every full example.
        prompts = dataset[preference_col_name]            # fast column access for HF datasets
        self.prompt_to_indices = defaultdict(list)
        for idx, b in enumerate(prompts):
            self.prompt_to_indices[b].append(idx)

        # sorted() -> deterministic, independent of any dataset iteration quirks
        self.prompts = sorted(self.prompt_to_indices.keys())

    def set_epoch(self, epoch):
        """
        This is called by Trainer internally - I have checked and it correctly updates the epoch
        """
        self.epoch = epoch

    def __iter__(self):
        # Local RNG seeded identically on every rank for this epoch.
        g = random.Random(self.seed + self.epoch)

        #print("-----Epoch-----")
        #print(self.epoch)
        #print('\n')

        all_batches = []
        for prompt in self.prompts:
            indices = self.prompt_to_indices[prompt].copy()
            if self.shuffle:
                g.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        if self.shuffle:
            g.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        total = 0
        for indices in self.prompt_to_indices.values():
            n = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size:
                n += 1
            total += n
        return total