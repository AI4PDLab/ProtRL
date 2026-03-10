"""
GRPO Diversity Study Script

Investigates whether LoRA causes diversity loss in GRPO training.
Tracks Hamming distance within and across generation groups.

Usage:
    python grpo_diversity_study.py --model_dir ../test/tiny --output_dir ./diversity_test
"""
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate.utils import set_seed
from trl import GRPOConfig, GRPOTrainer

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True, help="Path to the model")
parser.add_argument("--output_dir", type=str, default="./diversity_output", help="Output directory")
parser.add_argument("--target_length", type=int, default=50, help="Target sequence length")
parser.add_argument("--num_steps", type=int, default=100, help="Number of training steps")
parser.add_argument("--group_size", type=int, default=8, help="Number of generations per prompt")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

# Configuration
CONFIG = {
    "beta": 0.01,
    "learning_rate": 1e-5,
    "batch_size": 8,  # Must be divisible by num_generations (group_size)
    "num_epochs": 1,
}

os.makedirs(args.output_dir, exist_ok=True)

# Global diversity tracker
diversity_log = []
step_counter = [0]
all_sequences_this_step = []


def seed_everything(seed):
    """Sets seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def hamming_distance(s1, s2):
    """Compute Hamming distance between two sequences."""
    min_len = min(len(s1), len(s2))
    max_len = max(len(s1), len(s2))
    distance = sum(c1 != c2 for c1, c2 in zip(s1[:min_len], s2[:min_len]))
    distance += max_len - min_len  # Count length difference as mismatches
    return distance


def within_group_diversity(sequences):
    """Compute average pairwise Hamming distance within a group of sequences."""
    if len(sequences) < 2:
        return 0.0
    distances = [hamming_distance(s1, s2) for s1, s2 in combinations(sequences, 2)]
    return np.mean(distances)


def across_group_diversity(all_sequences):
    """Compute average Hamming distance across all sequences in a step."""
    if len(all_sequences) < 2:
        return 0.0
    # Sample if too many sequences
    if len(all_sequences) > 100:
        sampled = random.sample(all_sequences, 100)
    else:
        sampled = all_sequences
    distances = [hamming_distance(s1, s2) for s1, s2 in combinations(sampled, 2)]
    return np.mean(distances) if distances else 0.0


def reward_fn(completions, **kwargs):
    """
    Reward function combining:
    1. Length penalty (absolute deviation from target)
    2. Diversity tracking (logged, not part of reward)
    """
    global all_sequences_this_step
    
    rewards = []
    
    for completion in completions:
        # Length reward: penalize deviation from target length
        length_penalty = -abs(args.target_length - len(completion))
        rewards.append(float(length_penalty))
    
    # Track sequences for diversity computation
    all_sequences_this_step.extend(completions)
    
    # Every group_size completions, we have a full group
    if len(all_sequences_this_step) >= args.group_size:
        # Compute within-group diversity for the last group
        group = all_sequences_this_step[-args.group_size:]
        within_div = within_group_diversity(group)
        
        # Compute across-group diversity
        across_div = across_group_diversity(all_sequences_this_step)
        
        step_counter[0] += 1
        diversity_log.append({
            "step": step_counter[0],
            "within_group_diversity": within_div,
            "across_group_diversity": across_div,
            "num_sequences": len(all_sequences_this_step),
            "mean_length": np.mean([len(s) for s in group]),
            "unique_sequences": len(set(group)),
        })
        
        # Print progress
        print(f"Step {step_counter[0]}: within_div={within_div:.2f}, across_div={across_div:.2f}, unique={len(set(group))}/{args.group_size}")
        
        # Reset for next batch of steps
        if len(all_sequences_this_step) > 1000:
            all_sequences_this_step = all_sequences_this_step[-500:]
    
    return rewards


def generate_dummy_dataset(num_samples=500, prompt="M"):
    """Generate dummy dataset for online GRPO."""
    rows = [{"prompt": prompt, "completion": ""} for _ in range(num_samples)]
    return Dataset.from_list(rows)


# Main
seed_everything(args.seed)

# Create dataset
dataset = generate_dummy_dataset()
split = dataset.train_test_split(test_size=0.1, seed=args.seed, shuffle=True)
train_dataset = split['train']
eval_dataset = split['test']

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_dir,
    add_eos_token=True,
    add_bos_token=False,
    use_fast=True
)

# Ensure pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(args.model_dir)

print(f"Model: {args.model_dir}")
print(f"Target length: {args.target_length}")
print(f"Group size: {args.group_size}")
print(f"Output dir: {args.output_dir}")

# Training config
training_args = GRPOConfig(
    output_dir=args.output_dir,
    logging_steps=10,
    beta=CONFIG["beta"],
    num_train_epochs=CONFIG["num_epochs"],
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size"],
    do_train=True,
    do_eval=False,
    save_strategy="steps",
    save_total_limit=2,
    save_steps=50,
    num_generations=args.group_size,
    max_completion_length=args.target_length + 20,
    max_steps=args.num_steps,
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# Train
print("Starting training...")
trainer.train()

# Save model
trainer.save_model()

# Save diversity metrics
diversity_df = pd.DataFrame(diversity_log)
diversity_df.to_csv(os.path.join(args.output_dir, "diversity_metrics.csv"), index=False)

print(f"\nTraining complete!")
print(f"Model saved to: {args.output_dir}")
print(f"Diversity metrics saved to: {os.path.join(args.output_dir, 'diversity_metrics.csv')}")

# Print summary
if len(diversity_log) > 0:
    print("\n=== Diversity Summary ===")
    print(f"Initial within-group diversity: {diversity_log[0]['within_group_diversity']:.2f}")
    print(f"Final within-group diversity: {diversity_log[-1]['within_group_diversity']:.2f}")
    print(f"Initial unique sequences: {diversity_log[0]['unique_sequences']}/{args.group_size}")
    print(f"Final unique sequences: {diversity_log[-1]['unique_sequences']}/{args.group_size}")

torch.cuda.empty_cache()
