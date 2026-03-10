import argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr, pearsonr, linregress
import os

# --- CONFIGURATION ---
PREFIX_TOKEN = "" 
SUFFIX_TOKEN = ""
# ---------------------

def calculate_log_likelihood(model, tokenizer, sequence, device):
    full_sequence = PREFIX_TOKEN + str(sequence) + SUFFIX_TOKEN
    inputs = tokenizer(full_sequence, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss.item()
    return -loss 

def plot_and_correlate(df, x_col, y_col, x_label, y_label, output_prefix):
    # Filter valid data
    data = df.dropna(subset=[x_col, y_col])
    if len(data) < 2:
        print(f"Not enough data to plot {y_label} vs {x_label}")
        return

    # Calculate correlations
    s_corr, s_pval = spearmanr(data[x_col], data[y_col])
    p_corr, p_pval = pearsonr(data[x_col], data[y_col])
    
    # Linear Regression on Log10(Kd)
    # Since x-axis is log scale, we fit y ~ log10(x)
    log_x = np.log10(data[x_col])
    slope, intercept, r_value, p_value, std_err = linregress(log_x, data[y_col])
    r_squared = r_value ** 2
    
    print(f"{y_label} vs {x_label}: Spearman={s_corr:.3f} (p={s_pval:.2e}), Pearson={p_corr:.3f} (p={p_pval:.2e}), R2={r_squared:.3f}")

    # Plot
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=data, x=x_col, y=y_col, alpha=0.7)
    
    # Plot Regression Line
    # Create x points for line
    x_min, x_max = data[x_col].min(), data[x_col].max()
    x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    y_fit = slope * np.log10(x_fit) + intercept
    plt.plot(x_fit, y_fit, color='blue', linestyle='-', linewidth=1.5, label=f'Fit (R²={r_squared:.2f})')
    
    # Add zero line if plotting implicit reward
    if "implicit_reward" in y_col:
        plt.axhline(0, color='red', linestyle='--', linewidth=1, label="Reward = 0")
        
        # Stats for non-binders/weak binders (Kd >= 1e-4)
        non_binders = data[data[x_col] >= 1e-4]
        if not non_binders.empty:
            mean_reward = non_binders[y_col].mean()
            print(f"Mean {y_label} for Kd >= 1e-4 (n={len(non_binders)}): {mean_reward:.4f}")
            below_zero = (non_binders[y_col] < 0).sum()
            print(f"Non-binders with {y_label} < 0: {below_zero}/{len(non_binders)} ({below_zero/len(non_binders)*100:.1f}%)")
        
        # Stats for binders
        binders = data[data[x_col] < 1e-4]
        if not binders.empty:
             print(f"Mean {y_label} for Kd < 1e-4 (n={len(binders)}): {binders[y_col].mean():.4f}")

    plt.legend()
    plt.xscale('log')
    # Add p-values to title
    plt.title(f'{y_label} vs {x_label}\nSpearman: {s_corr:.2f} (p={s_pval:.1e}), Pearson: {p_corr:.2f} (p={p_pval:.1e})')
    plt.xlabel('Kd (Log Scale)')
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_{y_col}_vs_kd.png", dpi=300)
    plt.close()

def plot_binder_comparison(df, output_prefix):
    # Categorize
    df = df.copy()
    df['Category'] = df['kd'].apply(lambda x: 'Binder' if x < 1e-4 else 'Non-Binder')
    
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x='Category', y='implicit_reward', palette="Set2")
    sns.stripplot(data=df, x='Category', y='implicit_reward', color='black', alpha=0.5, jitter=True)
    
    plt.title('Implicit Reward: Binders vs Non-Binders')
    plt.ylabel('Implicit Reward')
    plt.xlabel('Category')

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_binder_vs_nonbinder_boxplot.png", dpi=300)
    plt.close()
    print(f"Saved box plot to {output_prefix}_binder_vs_nonbinder_boxplot.png")

def main():
    parser = argparse.ArgumentParser(description="Calculate log-likelihood of sequences using two LLMs for All Rounds data.")
    parser.add_argument("--ref_model", type=str, default="/users/nferruz/fstocco/Desktop/EGFR_analysis/FT_egf_zymctrl")
    parser.add_argument("--aligned_model", type=str, default="/users/nferruz/fstocco/Desktop/EGFR_analysis/output2_EGFR_round3.3")
    parser.add_argument("--base_dir", type=str, default="/users/nferruz/fstocco/Desktop/EGFR_analysis/EGFR_finalresults_december")
    parser.add_argument("--input_file", type=str, default="EGFR_test_all_rounds_summary.csv")
    parser.add_argument("--output_file", type=str, default="EGFR_all_rounds_scored.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only_plot", action='store_true', help="Skip scoring and only run plotting/analysis on existing output file.")
    args = parser.parse_args()

    # File paths
    input_path = os.path.join(args.base_dir, args.input_file)
    output_path = os.path.join(args.base_dir, args.output_file)

    if args.only_plot:
        print(f"Skipping scoring. Loading data from {output_path}...")
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            # Ensure sequence is string if we need it, though we mainly need scores now
            # We skip the deduplication/filling LOGIC usually done before scoring, 
            # assuming the output file is the final processed result.
            # However, if we want to be safe, we can re-apply some fillna if needed.
            # But usually output file has 'implicit_reward' already.
            
            # Re-apply KD numeric conversion just in case
            df['kd'] = pd.to_numeric(df['kd'], errors='coerce')
        else:
            print(f"Error: {output_path} not found. Cannot run plot-only mode.")
            return
            
        unique_df = df # Assume output file is already deduped or we accept it as is
        print(f"Loaded {len(unique_df)} rows for analysis.")

    else:
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows.")

        # 1. Handle Kd
        # Convert 'null' strings to NaN, coerce errors
        df['kd'] = pd.to_numeric(df['kd'], errors='coerce')
        
        # Track missing BEFORE filling
        df['is_missing_kd'] = df['kd'].isna()
        missing_count = df['is_missing_kd'].sum()
        print(f"Found {missing_count} rows with missing Kd.")

        # Fill for the 'filled' version of analysis
        df['kd'] = df['kd'].fillna(1e-4)

        # Deduplicate
        # We want to prioritize existing Kd over filled Kd if sequence is same?
        # Sort so that is_missing_kd=False comes first
        df = df.sort_values(by='is_missing_kd', ascending=True)
        unique_df = df.drop_duplicates(subset='sequence', keep='first').copy()
        print(f"Processing {len(unique_df)} unique sequences.")

        # Device
        device = args.device
        print(f"Using device: {device}")

        # Tokenizer & Models
        tokenizer = AutoTokenizer.from_pretrained(args.ref_model)
        
        print("Computing Ref Model LLs...")
        ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model, device_map="auto" if device == "cuda" else None, torch_dtype="auto").to(device)
        ref_model.eval()
        unique_df['ref_log_likelihood'] = [calculate_log_likelihood(ref_model, tokenizer, s, device) for s in tqdm(unique_df['sequence'], desc="Ref Model")]
        del ref_model
        torch.cuda.empty_cache()

        print("Computing Aligned Model LLs...")
        aligned_model = AutoModelForCausalLM.from_pretrained(args.aligned_model, device_map="auto" if device == "cuda" else None, torch_dtype="auto").to(device)
        aligned_model.eval()
        unique_df['aligned_log_likelihood'] = [calculate_log_likelihood(aligned_model, tokenizer, s, device) for s in tqdm(unique_df['sequence'], desc="Aligned Model")]
        del aligned_model
        torch.cuda.empty_cache()



        # Implicit Reward
        unique_df['implicit_reward'] = unique_df['aligned_log_likelihood'] - unique_df['ref_log_likelihood']
        
        # Save
        unique_df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

    # Plotting & Correlations (Runs for both cases)
    # We use output_path to determine where to save plots
    out_prefix = output_path.replace(".csv", "")
    
    # 1. All Data (including imputed)
    print("--- Analysis: All Data (Imputed Kd=1e-4) ---")
    
    # Ensure is_missing_kd column is present if we loaded from file without it
    if 'is_missing_kd' not in unique_df.columns:
        # Re-derive if possible, otherwise assume valid? 
        # Actually if we loaded from file, 'kd' might be all filled.
        # But we need to distinguish for the "Valid Kd Only" section.
        # If we can't determine, we might skip the filtered section or just run all.
        # Let's try to reconstruct based on original input if possible? 
        # No, simpler: check if 'kd' is 1e-4 exactly? (Risky).
        # We will wrap the "Valid Kd Only" section in a check.
        pass

    plot_and_correlate(unique_df, 'kd', 'implicit_reward', 'Kd', 'Implicit Reward', out_prefix)
    plot_and_correlate(unique_df, 'kd', 'ref_log_likelihood', 'Kd', 'Ref Log Likelihood', out_prefix)
    plot_and_correlate(unique_df, 'kd', 'aligned_log_likelihood', 'Kd', 'Aligned Log Likelihood', out_prefix)
    plot_binder_comparison(unique_df, out_prefix)
    
    # 2. Filtered Data (No imputed)
    print("--- Analysis: Valid Kd Only (No Imputed) ---")
    if 'is_missing_kd' in unique_df.columns:
        valid_df = unique_df[~unique_df['is_missing_kd']]
        if len(valid_df) > 0:
            out_prefix_valid = out_prefix + "_no_imputed"
            plot_and_correlate(valid_df, 'kd', 'implicit_reward', 'Kd', 'Implicit Reward', out_prefix_valid)
            plot_and_correlate(valid_df, 'kd', 'ref_log_likelihood', 'Kd', 'Ref Log Likelihood', out_prefix_valid)
            plot_and_correlate(valid_df, 'kd', 'aligned_log_likelihood', 'Kd', 'Aligned Log Likelihood', out_prefix_valid)
            plot_binder_comparison(valid_df, out_prefix_valid)
        else:
            print("No valid Kd rows found after filtering.")
    else:
        print("Note: 'is_missing_kd' column not found (likely running in --only_plot mode on a file that didn't save it). Skipping filtered analysis.")

if __name__ == "__main__":
    main()
