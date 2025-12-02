import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate.utils import set_seed
from trl import GRPOConfig, GRPOTrainer

# Config
BASE_DIR = "/users/nferruz/fstocco/Desktop/ProtRL/benchmarking/results/20251127_201239"
MODEL_PATH = "/users/nferruz/fstocco/Desktop/ProtRL/test/tiny"
SEED = 42
NUM_SAMPLES = 500
SPLIT_PERCENT = 0.1
NUM_EPOCHS = 1
NUM_GENERATIONS = 8

# Grid search parameters
BETAS = [0.1, 0.001]
LRS = [2e-3, 2e-4, 2e-5]
IMPORTANCE_SAMPLING = ["sequence", "token"]

set_seed(SEED)
os.makedirs(BASE_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, add_bos_token=False)

def generate_dataset(num_samples=NUM_SAMPLES):
    rows = [{"prompt": "<|bos|>", "completion": ""} for _ in range(num_samples)]
    return Dataset.from_list(rows)

# Prepare dataset
dataset = generate_dataset()
split = dataset.train_test_split(test_size=SPLIT_PERCENT, seed=SEED, shuffle=True)
train_dataset = split['train']
eval_dataset = split['test']

# Grid search
for beta in BETAS:
    for lr in LRS:
        for importance_sampling in IMPORTANCE_SAMPLING:
            
            run_name = f"trl_GRPO_beta{beta}_lr{lr}_is{importance_sampling}"
            output_dir = os.path.join(BASE_DIR, run_name)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Running: {run_name}")
            print(f"{'='*60}\n")
            
            # CSV storage for this run
            results = []
            
            def reward_len(completions, **kwargs):
                """Reward function that also saves samples to CSV"""
                # Get current step from trainer state if available
                trainer_state = kwargs.get('trainer_state')
                for sample in completions:
                    results.append({
                        "name": run_name,
                        "sequence": sample.replace(" ", ""),
                        "length": len(sample.replace(" ", "")),
                        "iteration_num": trainer_state.global_step
                    })
                return [float(-abs(50 - len(sequence.replace(" ", "")))) for sequence in completions]
            
            # Load fresh model for each run
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
            
            # Training config
            training_args = GRPOConfig(
                output_dir=output_dir,
                logging_steps=100,
                beta=beta,
                num_train_epochs=NUM_EPOCHS,
                learning_rate=lr,
                do_train=True,
                do_eval=True,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                num_generations=NUM_GENERATIONS,
                importance_sampling_level=importance_sampling,
                seed=SEED,
            )
            
            # Trainer
            trainer = GRPOTrainer(
                model=model,
                reward_funcs=reward_len,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
            
            # Train
            trainer.train()
            
            # Save CSV
            df = pd.DataFrame(results)
            csv_path = os.path.join(output_dir, "logs.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(results)} samples to {csv_path}")
            
            # Cleanup
            del model
            del trainer
            torch.cuda.empty_cache()

print("\nGrid search completed!")