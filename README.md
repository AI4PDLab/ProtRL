# ProtRL: Reinforcement Learning for Protein Language Models
<div align="center">
    <img src="https://github.com/user-attachments/assets/b5040d0c-74de-4627-bd2a-3e6344326ef5" width="350" >
</div>

A Reinforcement Learning (RL) framework for autoregressive protein Language Models (pLMs).
Currently we have implemented the following algorithms:
- Weighted DPO (with optional IRPO regularisation)
- GRPO 
- REINFORCE

This is the repository for the paper [*Guiding Generative Protein Language Models with Reinforcement Learning*](https://arxiv.org/abs/2412.12979). For the scripts used in the paper check the branch (experiments_v3)

## Table of Content
- [What's New in v3](#whats-new-in-v3)
- [About ProtRL](#about-protrl)
- [Usage](#usage)
- [Installation](#installation)
- [Example](#example)
  - [Multi-GPU Training](#multi-gpu-training)
- [General Usage](#general-usage)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Citation](#citation)

## What's New in v3

Version 3 is a major architectural refactor that adds multi-GPU support and consolidates both trainers under a single shared base class.

### New unified base trainer

Both `ProtRL_GRPOTrainer` and `ProtRL_wDPOTrainer` now share a common `ProtRLBaseTrainer` (in `src/ProtRL_Trainer/ProtRL_BaseTrainer.py`), which is built on top of HuggingFace's `Trainer`. This provides a consistent interface, shared data pipeline, and common logging infrastructure for all RL algorithms.

### DeepSpeed and FSDP support

Multi-GPU training is now supported via DeepSpeed (ZeRO stages 1/2/3) and FSDP. The framework automatically:
- Sets `device_map=None` when DeepSpeed or multi-GPU is detected (required to avoid `device_map="auto"` conflicts)
- Wraps the reference model with `prepare_deepspeed` or `prepare_fsdp` so it is properly sharded alongside the policy model

Pass a standard DeepSpeed config via `TrainingArguments`:
```python
args = ProtRLTrainingArgument(
    output_dir="my_run",
    deepspeed="ds_config.json",
    per_device_train_batch_size=4,
)
```

### IRPO regularisation for wDPO

`ProtRL_wDPOTrainer` now supports an optional IRPO regularisation term (adapted from IRPO, Shi et al. 2024) that encourages the model to increase the likelihood of high-reward completions:

```python
from src.pLM_weightedDPO import ProtRL_wDPOTrainer, ProtRL_wDPOTrainingArgument

args = ProtRL_wDPOTrainingArgument(
    output_dir="my_run",
    IRPO_regularisation=True,
    IRPO_regulariser_coeff=0.05,  # alpha weight for the regulariser
)
```

### PEFT / LoRA support

Both trainers now support PEFT adapters. Pass a `peft_config` to train only the adapter weights:

```python
from peft import LoraConfig

peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
trainer = ProtRL_GRPOTrainer(..., peft_config=peft_config)
```

A dual-adapter setup (separate adapters for policy and reference) is also supported via `model_adapter_name` and `ref_adapter_name` in `ProtRLTrainingArgument`.

### Mixture-of-Experts support

If the model has `output_router_logits` in its config, the auxiliary routing loss is automatically detected and added to the RL loss with `router_aux_loss_coef`.

### Per-token log probabilities

`get_batch_logps` now returns per-token log probabilities (not per-sequence sums). This gives GRPO finer-grained advantage weighting and makes the wDPO log-ratio more informative. Prompt tokens are masked out via the labels (`-100`).

### Preference batch sampler

`PreferenceBatchSampler` (`src/ProtRL_Trainer/preference_sampler.py`) ensures that every training batch contains sequences that share the same prompt (i.e., a valid preference set). This is required for both the wDPO softmax-weight computation and GRPO advantage normalisation.

### Spearman correlation monitoring

Both trainers log two Spearman correlation metrics at every logging step:
- `i_reward_correlation`: correlation between the implicit reward (log-ratio) and the true reward
- `logp_correlation`: correlation between the policy log-probability and the true reward

These are computed across the full macro-batch (gathered over all GPUs) and are a reliable signal that the trainer is learning.

---

## Quick Start

To verify the installation and pipeline functionality, you can run a quick test using a tiny model:
```bash
bash ProtRL.sh --model_dir test
```
This will automatically generate a tiny Llama model in `test/tiny` and run the full RL loop.

## About ProtRL

ProtRL allows you to:

- [**Train offline**](#offline-training) on pre-existing experimental data.
- [**Train online**](#online-training) with custom scoring functions in an iterative loop.

Based on the HuggingFace `Trainer`, we have extended it to support:

1. Passing custom pre-scored datasets at each RL iteration
2. Weighted DPO with IRPO regularisation
3. Offline GRPO with per-token KL and advantage normalisation
4. Multi-GPU training via DeepSpeed and FSDP
5. PEFT / LoRA adapters

### Quickstart Example

```python
from src.ProtRL_Trainer import ProtRLTrainingArgument
from src.pLM_GRPO import ProtRL_GRPOTrainer

training_args = ProtRLTrainingArgument(output_dir="ZymCTRL-GRPO", logging_steps=10)

trainer = ProtRL_GRPOTrainer(
    model="AI4PD/ZymCTRL",
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

For wDPO:

```python
from src.pLM_weightedDPO import ProtRL_wDPOTrainer, ProtRL_wDPOTrainingArgument

training_args = ProtRL_wDPOTrainingArgument(
    output_dir="ZymCTRL-wDPO",
    logging_steps=10,
    beta=0.1,
    IRPO_regularisation=True,
)

trainer = ProtRL_wDPOTrainer(
    model="AI4PD/ZymCTRL",
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## Usage

Both trainers expect datasets in HuggingFace `Dataset` format with the following columns:

| Column | Description |
|---|---|
| `prompt` | Conditioning prefix (empty string `""` for unconditional generation) |
| `completion` | The generated protein sequence |
| `reward` | Numerical score — higher is better |

Example:
```python
{"prompt": "<EC:1.1.1.1>", "completion": "MKVL...", "reward": 0.82}
```

> **Note:** Weights and rewards are treated as "higher is better". If your scoring function is minimised (e.g., energy), multiply by -1 before passing to the trainer.

All sequences sharing the same `prompt` value are treated as a preference set and will be batched together by `PreferenceBatchSampler`. The `preference_col_name` argument (default `"prompt"`) can be changed if your grouping key is a different column.

### Offline training

Use `train_exp.py`, which expects a CSV file with columns:
- `prompt`: prompt if any (use `""` for unconditional)
- `sequence`: pre-formatted protein sequences
- `reward`: numerical score for each sequence

```bash
python train_exp.py --model_dir "AI4PD/ZymCTRL" --csv "training_data.csv"
```

### Online training

For complex pipelines — where you explicitly generate, externally score, and then train each iteration — use `ProtRL.sh`. The loop:

1. **`seq_gen.py`** — generates sequences from the current policy model and writes a FASTA file
2. **`dataset_gen.py`** — parses the FASTA + external scores and appends to `logs.csv`
3. **`train_exp.py`** (or `train.py`) — runs one offline RL update on the new data
4. **`plot.py`** — plots sequence length over iterations

```bash
bash ProtRL.sh --model_dir "AI4PD/ZymCTRL" --output_dir "my_experiment"
```

`ProtRL.sh` runs with `python3` from your `PATH` by default; override with `PYTHON_EXEC=/path/to/python bash ProtRL.sh ...` if you need a specific interpreter (e.g. a virtualenv not on `PATH`).

For straightforward online rewards (e.g., sequence length), you can also use the standard HuggingFace TRL GRPO trainer directly:

```python
from trl import GRPOConfig, GRPOTrainer

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="ZymCTRL-GRPO", logging_steps=10)
trainer = ProtRL_GRPOTrainer(
    model="AI4PD/ZymCTRL",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model()
```

For the original DPO algorithm, we recommend the Hugging Face DPO Trainer.

Weighted DPO loss functions were adapted from those first described in [Widatalla et al., 2024](https://www.biorxiv.org/content/10.1101/2024.05.20.595026v1.abstract). You can find detailed explanations for each loss function in the Methods section of the [paper](https://arxiv.org/abs/2412.12979).

## Installation

```bash
git clone https://github.com/AI4PDLab/ProtRL.git
cd ProtRL
pip install -r requirements.txt
```

## Example

### Quickstart Colab

`example/ProtRL_quickstart_colab.ipynb` is the fastest way to get started. It runs on a free Colab T4 GPU and walks through:

1. Loading `AI4PD/ProtGPT3-112M` with LoRA (only ~0.5% of parameters trained)
2. **(Optional)** Supervised fine-tuning (SFT) warm-up on a small sequence set
3. Offline GRPO RL loop to steer the model toward a target sequence length

### TinyLLaMA Length Reduction

The `example/tiny-llama` directory demonstrates decreasing sequence length to 50 amino acids using a TinyLLaMA model that can be run locally on a single GPU.

```bash
cd example/tiny-llama/GRPO
bash local_protRL.sh
```

This generates a TinyLLaMA model, runs RL training, and plots length reduction over iterations.

<div align="center">
    <img src="https://github.com/user-attachments/assets/f51583e4-9f90-4170-acab-a4473503fdf3" width="350">
</div>

### Carbonic Anhydrase Fold in ZymCTRL

We also provide a more complex example in `example/ZymCTRL-fold`, where the fold of carbonic anhydrase is progressively adapted over RL iterations. This requires ESMFold and a GPU with at least 80 GB of VRAM.

### Benchmarking and Testing

We provide tools for testing and benchmarking different RL algorithms and hyperparameters.

- **Tiny Model**: `src/setup_tiny_model.py` generates a tiny Llama model for fast local testing and debugging. For a fast check you can run `ProtRL.sh` with `--model_dir test` — it will automatically build the tiny model in `test/tiny` and run the training.
- **Benchmarking Suite**: `benchmarking/` contains scripts to run comprehensive benchmarks comparing `ProtRL_GRPO`, `ProtRL_wDPO`, `ProtRL_REINFORCE`, and `trl_GRPO` across various learning rates and betas.
  - Run: `bash benchmarking/run_benchmark.sh`
  - Plot: `python benchmarking/plot_benchmark.py`

### Multi-GPU Training

Ready-to-use [Accelerate](https://huggingface.co/docs/accelerate) configs live in `example/multi_gpu/`.

| Config | Topology | Backend |
|---|---|---|
| `1node_4gpu.yaml` | 1 node, 4 GPUs | plain multi-GPU |
| `2nodes_4gpu.yaml` | 2 nodes, 4 GPUs each | DeepSpeed |

#### Single node — 4 GPUs

```bash
accelerate launch \
    --config_file example/multi_gpu/1node_4gpu.yaml \
    train.py \
    --model_dir "AI4PD/ProtGPT3-112M" \
    --label M \
    --iteration_num 1 \
    --max_iteration_num 30 \
    --output_dir results/my_run
```

Override `num_processes` on the command line if you have fewer than 4 GPUs:

```bash
accelerate launch \
    --config_file example/multi_gpu/1node_4gpu.yaml \
    --num_processes 2 \
    train.py ...
```

#### Multi-node — 2 nodes × 4 GPUs (DeepSpeed)

Run the same command on **every node**, changing only `--machine_rank`. Node 0 is the coordinator.

```bash
# On node 0
accelerate launch \
    --config_file example/multi_gpu/2nodes_4gpu.yaml \
    --machine_rank 0 \
    --main_process_ip <NODE0_IP> \
    --main_process_port 6000 \
    --num_machines 2 \
    --num_processes 8 \
    train.py \
    --model_dir "AI4PD/ProtGPT3-112M" \
    --label M \
    --iteration_num 1 \
    --max_iteration_num 30 \
    --output_dir results/my_run

# On node 1
accelerate launch \
    --config_file example/multi_gpu/2nodes_4gpu.yaml \
    --machine_rank 1 \
    --main_process_ip <NODE0_IP> \
    --main_process_port 6000 \
    --num_machines 2 \
    --num_processes 8 \
    train.py ...
```

`--num_processes` is the **total** number of GPUs across all nodes (2 nodes × 4 GPUs = 8). To add a custom DeepSpeed ZeRO config, pass `deepspeed="ds_config.json"` inside `ProtRLTrainingArgument` in your training script — the base trainer will handle reference-model sharding automatically.


### Experiments

To reproduce the experiments of our paper, all scripts are in the `experiments/` folder. Each experiment was run on a single H100 GPU. The `.sh` scripts are designed to work on a SLURM-based cluster:

```bash
bash experiments/experiment_name.sh
# or
sbatch experiments/experiment_name.sh
```

Each experiment will produce, fold, and calculate statistics for each considered feature.

## Adding a New RL Algorithm

Adding a new training algorithm to ProtRL requires implementing a **single method** — everything else (data loading, preference batching, reference model, PEFT, DeepSpeed, multi-GPU, logging) is provided by the base class for free.

### Pattern

```python
from src.ProtRL_Trainer import ProtRLBaseTrainer, ProtRLTrainingArgument

class MyAlgorithmTrainer(ProtRLBaseTrainer):

    def get_batch_loss_metrics(self, model, inputs, train_eval="train"):
        # inputs keys: input_ids, attention_mask, labels, reward
        # self.get_batch_logps(logits, labels)  → per-token log-probs (prompt masked)
        # self.compute_ref_logits(inputs)        → reference model logits (no grad)
        # self.beta                              → KL coefficient from args

        policy_logits = model(input_ids=inputs["input_ids"],
                              attention_mask=inputs["attention_mask"]).logits
        per_token_logps = self.get_batch_logps(policy_logits, inputs["labels"])

        loss = my_loss_fn(per_token_logps, inputs["reward"])

        metrics = {"my_metric": loss.item()}
        return loss, metrics
```

That's it. The trainer is then used exactly like `ProtRL_GRPOTrainer` or `ProtRL_wDPOTrainer`.

### Algorithms shipped with ProtRL

| File | Class | Algorithm | Notes |
|---|---|---|---|
| `src/pLM_GRPO.py` | `ProtRL_GRPOTrainer` | Offline GRPO | Per-token KL + advantage weighting |
| `src/pLM_weightedDPO.py` | `ProtRL_wDPOTrainer` | Weighted DPO | Softmax reward weights + optional IRPO |
| `src/pLM_REINFORCE.py` | `ProtRL_REINFORCETrainer` | REINFORCE | Simplest baseline; optional entropy bonus |

### Example: REINFORCE in 5 lines

```python
from src.pLM_REINFORCE import ProtRL_REINFORCETrainer, ProtRL_REINFORCETrainingArgument

args = ProtRL_REINFORCETrainingArgument(
    output_dir="my_run",
    entropy_bonus=0.01,   # optional: encourage exploration
)

trainer = ProtRL_REINFORCETrainer(
    model="AI4PD/ZymCTRL",
    processing_class=tokenizer,
    args=args,
    train_dataset=dataset,
)
trainer.train()
```

## Notes

- `seq_gen.py` generates a FASTA file in the format: `>name \t perplexity \n sequence`
- Ranked DPO has been discontinued — it is theoretically always outperformed by weighted DPO
- `ProtRLBaseTrainer` always sets `dataloader_drop_last=True` to ensure Spearman correlation metrics are reliable. Batch sizes below 3 will trigger a warning.
- When using DeepSpeed ZeRO-3, `create_reference_model` will raise a `ValueError` — use a separate checkpoint as `ref_model` instead.

## Troubleshooting

Feel free to contribute or raise issues if you encounter any problems! We are working to make the framework more accessible and detailed.

## Work in Progress

- [x] LoRA + SFT + RL example — see `example/ProtRL_quickstart_colab.ipynb`

## References

- ESM1v: "Language models enable zero-shot prediction of the effects of mutations on protein function" Joshua Meier et al.; doi: https://doi.org/10.1101/2021.07.09.450648
- ProteinMPNN: "Robust deep learning–based protein sequence design using ProteinMPNN", J. Dauparas et al. Science 378, 49-56 (2022). DOI:10.1126/science.add2187
- CLEAN: "Enzyme function prediction using contrastive learning". Science 379, 1358-1363 (2023). DOI:10.1126/science.adf2465
- IRPO: "Reinforcement Learning from Human Feedback with Active Queries", Shi et al. 2024

## Citation

If you use ProtRL, please cite our [preprint](https://arxiv.org/abs/2412.12979):

```
@misc{stocco2024guidinggenerativeproteinlanguage,
      title={Guiding Generative Protein Language Models with Reinforcement Learning}, 
      author={Filippo Stocco and Maria Artigues-Lleixa and Andrea Hunklinger and Talal Widatalla and Marc Guell and Noelia Ferruz},
      year={2024},
      eprint={2412.12979},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2412.12979}, 
}
```
