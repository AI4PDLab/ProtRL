# ProtRL: Reinforcement Learning for Protein Language Models
<div align="center">
    <img src="https://github.com/user-attachments/assets/b5040d0c-74de-4627-bd2a-3e6344326ef5" width="350" >
</div>

A Reinforcement Learning (RL) framework for autoregressive protein Language Models (pLMs).
Currently we have implemented the following algorithms: 
- Weighted DPO
- GRPO (```bnpo```, ```dr_grpo``` and ```grpo```)

This is the repository for the paper [*Guiding Generative Protein Language Models with Reinforcement Learning*](https://arxiv.org/abs/2412.12979). 

## Table of Content
- [About ProtRL](#about-protrl)
- [Usage](#usage)
- [Installation](#installation)
- [Example](#example)
- [General Usage](#generalusage)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Citation](#citation)

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

Based on the GRPO implementation in [Hugging Face’s TRL library](https://huggingface.co/docs/trl/main/en/grpo_trainer), we have extended the trainer to support:

1. Passing custom datasets at each iteration  
2. Weighted variant of DPO (not available in the standard Hugging Face trainer)

### Quickstart Example

```python
from src.utils import *
from src.pLM_weigtedDPO import weighted_DPO
from src.pLM_GRPO import pLM_GRPOTrainer
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(output_dir="ZymCTRL-wDPO", logging_steps=10)

trainer = pLM_wDPOTrainer( #pLM_rDPOTrainer, pLM_GRPOTrainer
    model= "AI4PD/ZymCTRL",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
```
## Usage
Trainer accepts the datasets in a HF standard format, for example: 
```python
{"prompt": "The sky is", "completion": " blue.", "advantage":10}
```
### Offline training
Use ```train_exp.py```, which expects a CSV file with columns:
- prompt: prompt if any (in case of conditional generation)
- sequence: pre-formatted protein sequences
- advantage: numerical weight for each sequence
  
```python 
python train_exp.py --model_dir "AI4PD/ZymCTRL" --csv "training_data.csv"
```
the code will generate the dataset for you and train your model. 

### Online training
1. We reccomend using the HF implementation of GRPO for straightforward rewards (e.g., sequence length, amino-acid ratios), use the standard GRPO trainer:

```python 
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("your_dataset")
split = dataset.train_test_split(test_size=0.80, seed=42, shuffle=True)

train_dataset = split['train']
eval_dataset   = split['test']

# Define the reward function, in this case
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

tokenizer = AutoTokenizer.from_pretrained("AI4PD/ZymCTRL")
tokenizer.padding_side = "left"
tokenizer.eos_token_id = 1
tokenizer.pad_token_id = 0

training_args = GRPOConfig(output_dir="ZymCTRL-GRPO", logging_steps=10)

trainer = GRPO_trainer(
    model= "AI4PD/ZymCTRL",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model() 
```
For complex pipelines—where you explicitly generate, save, and externally score sequences each iteration, you can use our  trainers. This is ideal for scoring in CPU arrays before training on GPU:

```python
from src.utils import *
from src.pLM_weigtedDPO import weighted_DPO
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(output_dir="ZymCTRL-GRPO", logging_steps=10)

trainer = weighted_DPO( #pLM_GRPOTrainer
    model= "AI4PD/ZymCTRL",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
```
> **_Note:_** The reward_funcs is ignored and can be set as a function always returning 0, see examples. 

For the original DPO algorithm, we recommend the Hugging Face DPO Trainer.

Weighted DPO loss functions were adapted from the firsts described in [Widatalla et al., 2024](https://www.biorxiv.org/content/10.1101/2024.05.20.595026v1.abstract). You can find detailed explanations for each loss function and its changes in formulation in the Methods section of the [paper](https://arxiv.org/abs/2412.12979).

> **_Note:_** Weights and advantages are treated as "the higher, the better." If your scoring function is designed to be minimized, please multiply it by -1.

## Installation

```bash
git clone https://github.com/AI4PDLab/ProtRL.git
cd ProtRL
pip install -r requirements.txt
```

## Example 
### TinyLLaMA Length Reduction
The example directory includes ```tiny-llama``` directory, which demonstrates decreasing sequence length to 50 amino acids using a TinyLLaMA model that can be run locally on a single GPU. 

```bash
cd example/GRPO
bash ProtRL-local.sh
```

This generates a TinyLLaMA model, runs RL training, and plots length reduction over iterations.
<div align="center">
    <img src="https://github.com/user-attachments/assets/f51583e4-9f90-4170-acab-a4473503fdf3" width="350">

</div>


### Carbonic Anhydrase Fold in ZymCTRL
We also provide a more complex example in ```example/ZymCTRL-fold```, where the fold of carbonic anhydrase is progressively adapted over RL iterations. In this case esm-fold is required and a GPU of 80GB. 

### Benchmarking and Testing
We provide tools for testing and benchmarking different RL algorithms and hyperparameters.
- **Tiny Model**: `setup_tiny_model.py` generates a tiny Llama model for fast local testing and debugging. For fast check you can just run ProtRL.sh and set the `--model_dir` to `test`. it will automatically build the tiny model in `test/tiny` and run the training.
- **Benchmarking Suite**: `benchmarking/` contains scripts to run comprehensive benchmarks comparing `pLM_GRPO`, `weighted_DPO`, and `trl_GRPO` across various learning rates and betas.
  - Run: `bash benchmarking/run_benchmark.sh`
  - Plot: `python benchmarking/plot_benchmark.py`

### Experiments

To reproduce the experiments of our paper, you can find all the scripts in the `experiments` folder. Given the size and computational needs of pLMs, each one of the experiments were executed in one H100 GPU, with differing times of execution. All the parameters and external data used in the experiments can be found in this repo. The `.sh` scripts can be executed from the same folder to conduct each experiment, they have been built to work on a SLURM based cluster, given the need of GPU-intensive computing. To reproduce the results run: 

```bash
bash experiment_name.sh
```
or 
```bash 
sbatch experiment_name.sh
```
Replace `experiment_name` with the desired experiment script path. Each experiment will produce, fold and calculate statistics for each considered feature.

## Notes
seq_gen.py in the main directory generates a fasta file with this format ```>fasta_name /t perplexity /t intrinsic_reward /n sequence```

We discontinue ranked DPO as theoretically it will always be outperformed by weighted DPO

## Troubleshooting

Please take a look at the documentation for more details on how to configure and run your experiments.

Feel free to contribute or raise issues if you encounter any problems! We are working to make it more accessible and detailed

## Work in Progress

[ ] LoRa example

## References

- ESM1v: "Language models enable zero-shot prediction of the effects of mutations on protein function" Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives; doi: https://doi.org/10.1101/2021.07.09.450648. Computed using https://github.com/seanrjohnson/protein_gibbs_sampler/
- ProteinMPNN: "Robust deep learning–based protein sequence design using ProteinMPNN", J. Dauparas et al. Science378,49-56(2022).DOI:10.1126/science.add2187
- CLEAN: "Enzyme function prediction using contrastive learning". Science379,1358-1363(2023). DOI:10.1126/science.adf2465, GitHub: "https://github.com/tttianhao/CLEAN?tab=readme-ov-file"

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

 


