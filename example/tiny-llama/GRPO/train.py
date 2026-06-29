import os, sys, argparse, torch
from datasets import Dataset
from transformers import AutoTokenizer

# ProtRL root is three directories up from this file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.ProtRL_Trainer import ProtRLTrainingArgument
from src.pLM_GRPO import ProtRL_GRPOTrainer

TARGET_LEN = 50  # amino acids

# ---------- helpers ----------

def parse_fasta(path):
    seqs, name = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split("\t")[0]
                seqs[name] = ""
            elif name:
                seqs[name] += line
    return seqs

def build_dataset(fasta_path, label):
    seqs = parse_fasta(fasta_path)
    rows = [
        {"prompt": label, "completion": seq, "reward": float(-abs(TARGET_LEN - len(seq)))}
        for seq in seqs.values() if seq
    ]
    return Dataset.from_list(rows)

# ---------- main ----------

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label",         type=str, required=True)
parser.add_argument("--model_dir",     type=str, required=True)
args = parser.parse_args()

root_dir = os.path.dirname(os.path.abspath(__file__))

fasta   = os.path.join(root_dir, "data", "inputs",
                        f"seq_gen_{args.label}_iteration{args.iteration_num - 1}.fasta")
dataset = build_dataset(fasta, args.label)
split   = dataset.train_test_split(test_size=0.2, seed=42)

# Load from previous output, or from the initial base model on the first training iteration
model_path = (
    os.path.join(root_dir, f"output_iteration{args.iteration_num - 1}")
    if args.iteration_num > 1 else args.model_dir
)

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(root_dir, "models", "tokenizer"), use_fast=True
)

training_args = ProtRLTrainingArgument(
    output_dir=os.path.join(root_dir, f"output_iteration{args.iteration_num}"),
    num_train_epochs=1,
    per_device_train_batch_size=8,
    beta=0.1,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
)

trainer = ProtRL_GRPOTrainer(
    model=model_path,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
)

trainer.train()
trainer.save_model()
torch.cuda.empty_cache()
