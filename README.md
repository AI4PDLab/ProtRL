# ProtRL: Reinforcement Learning for Protein Language Models
<div align="center">
    <img src="https://github.com/user-attachments/assets/b5040d0c-74de-4627-bd2a-3e6344326ef5" width="350" >
</div>

A Reinforcement Learning (RL) framework for autoregressive protein Language Models (pLMs).

This repository accompanies the paper [*Guiding Generative Protein Language Models with Reinforcement Learning*](https://arxiv.org/abs/2412.12979).

Currently supported algorithms:
- **GRPO** — Group Relative Policy Optimization
- **Weighted DPO** — Weighted Direct Preference Optimization (with optional IRPO regularisation)
- **REINFORCE** — with an optional entropy bonus

---

## 1. Installation

Set up the environment and install dependencies:

```bash
git clone https://github.com/AI4PDLab/ProtRL.git
cd ProtRL
pip install -r requirements.txt
```

---

## 2. Quickstart

A very simple demo can be run with a tiny model locally:
```bash
bash ProtRL.sh --model_dir "test" 
```

Additionally we provide two interactive notebooks to get you started quickly:

### 🧪 Production-Ready Protein Design Workflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I-nlMYFu6vHJcKopUWmbYN6yvxnkx3kj?usp=sharing)
 **[ProRL_csv_experimental.ipynb](ProRL_csv_experimental.ipynb)**
A complete, ready-to-use notebook for any protein design task. It allows you to feed back experimental data from a custom CSV to automatically fine-tune and reinforce the protein language model (SFT warm-up followed by GRPO reinforcement learning).
1. **Load data from a CSV**: Input your custom sequences and experimental rewards.
2. **SFT Warm-Up (with Train/Eval Split)**: Automatically filters and fine-tunes on sequences with rewards higher than the mean, using a standard Hugging Face `Trainer` with a causal-LM data collator on an 80/20 train/eval split with step-wise logging.
3. **ProtRL GRPO (with Train/Eval Split)**: Applies reinforcement learning on the complete dataset with step-wise metrics (the trainer tokenizes internally — pass raw sequences).
4. **Training Curves**: Plots training/evaluation loss curves and Spearman correlation tracking metrics using simple matplotlib charts.

###  Standard Toy Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YZvqI6PHfxckGlKaJtE4LDNWYn8gn3G_?usp=sharing)

 **[example/ProRL_example.ipynb](example/ProRL_example.ipynb)**
A self-contained toy workflow demonstrating mutation dataset generation, SFT, and GRPO length control.

---

## 3. Online Iterative RL Loop

For complex pipelines where you generate, score externally (e.g., structure, stability, or activity assays), and update the policy model iteratively:

```bash
bash ProtRL.sh --model_dir "AI4PD/ProtGPT3-112M" --output_dir "my_experiment"
```

The script automatically executes the following loop for each iteration:
1. **`seq_gen.py`**: Generates sequence completions from the active model checkpoint.
2. **`dataset_gen.py`**: Gathers generated sequences and prepares them for training.
3. **`train.py`**: Runs an offline RL update (e.g., GRPO) on the compiled dataset.
4. **`plot.py`**: Visualizes metrics such as sequence length across iterations.

For debugging or local testing without a large GPU, you can run:
```bash
bash ProtRL.sh --model_dir test
```
This automatically generates a tiny model locally and runs the online loop.

---

## 4. Output Results Directory Structure

Running the online loop creates a results directory `results/YYYYMMDD_HHMMSS/` (or a custom folder specified via `--output_dir`). The contents are organized as follows:

```
results_directory/
├── logs.csv                         # Central database of all generated sequences over iterations
├── length_over_iterations.png       # Line plot tracking sequence length over training progress
├── seq_gen_<label>_iteration1.fasta  # Raw generated sequences from iteration 1
├── seq_gen_<label>_iteration2.fasta  # Raw generated sequences from iteration 2
├── output_iteration1/               # Fine-tuned policy model checkpoint after iteration 1
└── output_iteration2/               # Fine-tuned policy model checkpoint after iteration 2
```

---

## 5. Offline One-Shot Training

If you already have a pre-existing CSV dataset containing columns `prompt`, `sequence`, and `reward`, you can train the policy model directly:

```bash
python train_exp.py --model_dir "AI4PD/ProtGPT3-112M" --csv "my_dataset.csv"
```

For ProtGPT3 the `prompt` column must be the direction token (`"1"` forward / `"2"` reverse),
and `sequence` the raw amino-acid completion — see the ProtGPT3 prompting note in
[§7 Dataset Format](#7-dataset-format). `example/fake_dataset.csv` is a ready-to-use example.

---

## 6. Trainer API Usage

Initialize the trainers in Python using the standard Hugging Face Trainer interface:

### GRPO
```python
from src.ProtRL_Trainer import ProtRLTrainingArgument
from src.pLM_GRPO import ProtRL_GRPOTrainer

training_args = ProtRLTrainingArgument(output_dir="ProtGPT3-GRPO", logging_steps=10)

trainer = ProtRL_GRPOTrainer(
    model="AI4PD/ProtGPT3-112M",
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

### Weighted DPO
```python
from src.pLM_weightedDPO import ProtRL_wDPOTrainer, ProtRL_wDPOTrainingArgument

training_args = ProtRL_wDPOTrainingArgument(
    output_dir="ProtGPT3-wDPO",
    logging_steps=10,
    beta=0.1,
    IRPO_regularisation=True,
)

trainer = ProtRL_wDPOTrainer(
    model="AI4PD/ProtGPT3-112M",
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

### REINFORCE
```python
from src.pLM_REINFORCE import ProtRL_REINFORCETrainer, ProtRL_REINFORCETrainingArgument

training_args = ProtRL_REINFORCETrainingArgument(
    output_dir="ProtGPT3-REINFORCE",
    logging_steps=10,
    entropy_bonus=0.0,  # set > 0 to encourage exploration
)

trainer = ProtRL_REINFORCETrainer(
    model="AI4PD/ProtGPT3-112M",
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

---

## 7. Dataset Format

The trainers expect Hugging Face `Dataset` inputs containing:
* `prompt`: The conditioning tag, which varies depending on the model and target task.
  * E.g. `"1"` for ProtGPT3 — its **direction token** (`"1"` = forward N→C, `"2"` = reverse). ProtGPT3 was trained to condition on this, so it is the correct prompt (not `"M"`).
  * E.g. `"<EC:1.1.1.1>"` or other specific function tags for guided protein language models (like ZymCTRL).
* `completion`: The **raw** amino-acid sequence, e.g. `"HGEGTFTSDLSKQME"`.
* `reward`: A numerical score (higher is better).

Example entry:
```json
{"prompt": "1", "completion": "HGEGTFTSDLSKQME", "reward": 1.0}
```

> ⚠️ **ProtGPT3 prompting — get these two things right or generation degenerates:**
> 1. **Prompt with the direction token** (`"1"` forward / `"2"` reverse), not `"M"`.
> 2. **Add a leading BOS token.** For generation load the tokenizer with `add_bos_token=True`
>    (`AutoTokenizer.from_pretrained(model, add_bos_token=True, add_eos_token=False)`); the RL
>    trainer adds BOS to the prompt for you. Generating from a BOS-less / direction-token-less
>    prompt drives the model out-of-distribution and yields repetitive junk like `MMMKKK...GGGG`.
>
> ⚠️ **Pass the raw sequence — do not insert spaces between residues.**
> ProtGPT3 (and the other character-level pLMs used here) tokenize **one token per residue**
> and have no space token. The trainer tokenizes the `completion` for you, so a space-separated
> string like `"H G E G T"` would map every space to `[UNK]`, silently breaking the RL signal.
> When you decode generated sequences the tokenizer *displays* spaces between residues; strip
> them with `.replace(" ", "")` before storing or reusing them.

---

## Citation

If you use ProtRL in your research, please cite:

```bibtex
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
