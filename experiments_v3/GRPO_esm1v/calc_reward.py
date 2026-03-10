import argparse
import os
import sys
import pandas as pd
import torch

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import esm
except ImportError:
    pass

from utils import compute_esm1v_pll

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """
    Calculates the reward for the GRPO_esm1v experiment.
    Reward is defined as the ESM-1v Pseudo-Log-Likelihood (PLL).
    PLL is calculated by masking each residue and summing the log-probabilities.
    Maximizing PLL <=> Maximizing likelihood of the sequence under the ESM-1v model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True, help="Current iteration number")
    parser.add_argument("--label", type=str, required=True, help="Label for the experiment (unused but kept for compatibility)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory containing logs and results")
    args = parser.parse_args()

    logs_file = os.path.join(args.output_dir, "logs.csv")
    
    # 1. Validation: Check if logs file exists
    if not os.path.exists(logs_file):
        print(f"Error: Logs file not found at {logs_file}")
        return

    # Model path - hardcoded as per previous context or passed via args?
    # Keeping it hardcoded relative path for now as per previous implementation plan
    ESM1V_MODEL_PATH = "../../../models/esm1v_t33_650M_UR90S_1.pt" 
    
    if not os.path.exists(ESM1V_MODEL_PATH):
        print(f"Error: ESM-1v model not found at {ESM1V_MODEL_PATH}")
        return

    try:
        df = pd.read_csv(logs_file)
    except Exception as e:
        print(f"Error reading logs file: {e}")
        return
    
    # Filter for current iteration to process only relevant rows
    mask = df['iteration_num'] == args.iteration_num
    
    if not mask.any():
        print(f"Warning: No rows found for iteration {args.iteration_num}")
        return

    print(f"Processing rewards for iteration {args.iteration_num}...")

    # 2. Extract Sequences
    indices_to_process = []
    raw_sequences = []
    
    for index, row in df[mask].iterrows():
        sequence = row['sequence']
        # Extract raw sequence assuming format <label><sep><start>SEQUENCE<end>...
        raw_seq = sequence
        if "<start>" in sequence:
            try:
                raw_seq = sequence.split("<start>")[1].split("<end>")[0]
            except Exception:
                pass # Keep original if parse fails
        
        indices_to_process.append(index)
        raw_sequences.append(raw_seq)

    if not raw_sequences:
        print("Warning: No sequences to process.")
        return

    # 3. Load ESM-1v Model
    print("Loading ESM-1v Model...")
    try:
        # Use safe_globals to avoid pickle errors with new PyTorch versions
        with torch.serialization.safe_globals([argparse.Namespace]):
            model_esm, alphabet = esm.pretrained.load_model_and_alphabet_local(ESM1V_MODEL_PATH)
        
        batch_converter = alphabet.get_batch_converter()
        model_esm.eval()
        model_esm.to(device)
        print("Model loaded successfully.")
        
    except Exception as e:
        print(f"Error loading ESM-1v model: {e}")
        return

    # 4. Compute PLL
    print(f"Calculating ESM-1v PLL for {len(raw_sequences)} sequences... (This may be slow)")
    try:
        # Note: compute_esm1v_pll is defined in utils.py
        # It handles the O(L^2) masking and forward passes.
        pll_scores = compute_esm1v_pll(model_esm, batch_converter, raw_sequences, device)
        
        # 5. Update DataFrame
        updates_count = 0
        for i, index in enumerate(indices_to_process):
            score = pll_scores[i]
            
            # Log the raw PLL score
            df.at[index, 'esm1v_pll'] = score
            
            # Reward: Maximize PLL
            df.at[index, 'reward'] = score
            updates_count += 1
            
        print(f"Computed PLL for {updates_count} sequences.")
        
    except Exception as e:
        print(f"Error computing PLL: {e}")
        return

    # 6. Save Results
    try:
        df.to_csv(logs_file, index=False)
        print("Successfully saved updated logs.")
    except Exception as e:
        print(f"Error saving updated logs file: {e}")

if __name__ == "__main__":
    main()
