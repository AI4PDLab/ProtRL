# ProtRL: Reinforcement Learning for Protein Language Models
<div align="center">
    <img src="https://github.com/user-attachments/assets/b5040d0c-74de-4627-bd2a-3e6344326ef5" width="350" >
</div>

A Reinforcement Learning (RL) framework for autoregressive protein Language Models (pLMs).
Currently we have implemented the following algorithms:
- Weighted DPO (with optional IRPO regularisation)
- GRPO 
- REINFORCE

This repository accompanies the paper [*Guiding Generative Protein Language Models with Reinforcement Learning*](https://arxiv.org/abs/2412.12979).

Currently supported algorithms:
- **GRPO** (Group Relative Policy Optimization)
- **Weighted DPO** (Weighted Direct Preference Optimization)
- **REINFORCE**

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
 **[ProRL_csv_experimental.ipynb](file:///users/nferruz/fstocco/Desktop/ProtRL/ProRL_csv_experimental.ipynb)**
A complete, ready-to-use notebook for any protein design task. It allows you to feed back experimental data from a custom CSV to automatically fine-tune and reinforce the protein language model (SFT warm-up followed by GRPO reinforcement learning).
1. **Load data from a CSV**: Input your custom sequences and experimental rewards.
2. **SFT Warm-Up (with Train/Eval Split)**: Automatically filters and fine-tunes on sequences with rewards higher than the mean, using TRL's `SFTTrainer` (no manual tokenization needed) on an 80/20 train/eval split with step-wise logging.
3. **ProtRL GRPO (with Train/Eval Split)**: Applies reinforcement learning on the complete dataset with step-wise metrics (no manual tokenization needed).
4. **Training Curves**: Plots training/evaluation loss curves and Spearman correlation tracking metrics using simple matplotlib charts.

###  Standard Toy Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YZvqI6PHfxckGlKaJtE4LDNWYn8gn3G_?usp=sharing)

 **[example/ProRL_example.ipynb](file:///users/nferruz/fstocco/Desktop/ProtRL/example/ProRL_example.ipynb)**
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

---

## 7. Dataset Format

The trainers expect Hugging Face `Dataset` inputs containing:
* `prompt`: The conditioning tag, which varies depending on the model and target task.
  * E.g. `"M"` for ProtGPT3 (representing the starting Methionine amino acid).
  * E.g. `""` (empty string) for unconditional generation.
  * E.g. `"<EC:1.1.1.1>"` or other specific function tags for guided protein language models (like ZymCTRL).
* `completion`: The generated amino acid sequence.
* `reward`: A numerical score (higher is better).

Example entry:
```json
{"prompt": "M", "completion": "H G E G T F T S D L S K Q M E", "reward": 1.0}
```

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
