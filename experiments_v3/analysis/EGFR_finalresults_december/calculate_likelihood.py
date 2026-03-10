import argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import os

# --- CONFIGURATION ---
PREFIX_TOKEN = "" 
SUFFIX_TOKEN = ""
colors = ['#F8BA00','#4F1AE1','#7EC2C5']
# ---------------------

# Style configuration
plt.figure(figsize=(10, 6))
font = {'family': 'sans-serif', 'size': 20}
plt.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2.5


def calculate_log_likelihood(model, tokenizer, sequence, device):
    full_sequence = PREFIX_TOKEN + str(sequence) + SUFFIX_TOKEN
    inputs = tokenizer(full_sequence, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss.item()
    return -loss 

def plot_and_correlate(df, x_col, y_col, x_label, y_label, output_prefix, color=None):
    # Filter valid data
    data = df.dropna(subset=[x_col, y_col])
    if len(data) < 2:
        print(f"Not enough data to plot {y_label} vs {x_label}")
        return

    # Calculate correlations
    s_corr, s_pval = spearmanr(data[x_col], data[y_col])
    p_corr, p_pval = pearsonr(data[x_col], data[y_col])
    print(f"{y_label} vs {x_label}: Spearman={s_corr:.3f} (p={s_pval:.2e}), Pearson={p_corr:.3f} (p={p_pval:.2e})")

    # Plot
    plt.figure(figsize=(6, 5))
    # Use regplot to show correlation line. logx=True since x-axis (Kd) is log-scaled contextually
    sns.regplot(data=data, x=x_col, y=y_col, logx=True, 
                color=color if color else colors[0], 
                scatter_kws={'alpha': 0.7},
                line_kws={'color': 'black', 'label': f'Fit (Pearson: {p_corr:.2f})'}
               )
    
    # Add zero line if plotting implicit reward
    if "implicit_reward" in y_col:
        #plt.axhline(0, color='red', linestyle='--', linewidth=1, label="Reward = 0")
        
        # Stats for non-binders/weak binders (Kd >= 1e-4)
        non_binders = data[data[x_col] >= 1e-4]
        if not non_binders.empty:
            mean_reward = non_binders[y_col].mean()
            print(f"Mean {y_label} for Kd >= 1e-4 (n={len(non_binders)}): {mean_reward:.4f}")
            below_zero = (non_binders[y_col] < 0).sum()
            print(f"Non-binders with {y_label} < 0: {below_zero}/{len(non_binders)} ({below_zero/len(non_binders)*100:.1f}%)")

    #plt.legend()
    plt.xscale('log')
    plt.title(f'{y_label} vs {x_label}\nSpearman: {s_corr:.2f}, Pearson: {p_corr:.2f}')
    plt.xlabel('Kd (Log Scale)')
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_{y_col}_vs_kd.pdf", dpi=600)
    plt.close()

def plot_binder_comparison(df, output_prefix):
    # Categorize
    df = df.copy()
    df['Category'] = df['kd'].apply(lambda x: 'Binder' if x < 1e-4 else 'Non-Binder')
    
    binders = df[df['Category'] == 'Binder']['implicit_reward']
    non_binders = df[df['Category'] == 'Non-Binder']['implicit_reward']
    
    if len(binders) == 0 or len(non_binders) == 0:
        print("Cannot compute stats: one category is empty.")
        return

    # Statistics
    mean_diff = binders.mean() - non_binders.mean()
    stat, pval = mannwhitneyu(binders, non_binders, alternative='two-sided')
    
    print(f"Binder vs Non-Binder: Mean Diff={mean_diff:.4f}, MWU p-value={pval:.2e}")

    plt.figure(figsize=(6, 6))
    # Boxplot light gray
    sns.boxplot(data=df, x='Category', y='implicit_reward', color='lightgray', showfliers=False)
    # Dots colored
    sns.stripplot(data=df, x='Category', y='implicit_reward', hue='Category', s=15, palette=colors[:2], alpha=0.7, jitter=True, legend=False)
    
    # Annotate stats
    y_max = df['implicit_reward'].max()
    y_range = y_max - df['implicit_reward'].min()
    
    plt.text(0.5, y_max + 0.05 * y_range, f"Mean Diff: {mean_diff:.3f}", ha='center', fontsize=12)
    plt.text(0.5, y_max + 0.12 * y_range, f"p-value: {pval:.2e}", ha='center', fontsize=12)
    
    plt.title('Implicit Reward: Binders vs Non-Binders')
    plt.ylabel('Implicit Reward')
    plt.xlabel('Category')
    plt.ylim(top=y_max + 0.2 * y_range) # Add space for text
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_binder_vs_nonbinder_boxplot.pdf", dpi=600)
    plt.close()
    print(f"Saved box plot to {output_prefix}_binder_vs_nonbinder_boxplot.pdf")

def main():
    parser = argparse.ArgumentParser(description="Calculate log-likelihood of sequences using two LLMs.")
    parser.add_argument("--ref_model", type=str, default="/users/nferruz/fstocco/Desktop/EGFR_analysis/FT_egf_zymctrl")
    parser.add_argument("--aligned_model", type=str, default="/users/nferruz/fstocco/Desktop/EGFR_analysis/output2_")
    parser.add_argument("--base_dir", type=str, default="/users/nferruz/fstocco/Desktop/EGFR_analysis/EGFR_finalresults_december")
    parser.add_argument("--output_file", type=str, default="EGFR_merged_scored.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # File paths
    file_all = os.path.join(args.base_dir, "EGFR_test_all_rounds (Competition Buffer)_summary.csv")
    file_r3 = os.path.join(args.base_dir, "EGFR_round3_summary.csv")
    file_r32 = os.path.join(args.base_dir, "EGFR_round3.2_summary.csv")

    print("Loading data...")
    # Load with low_memory=False to avoid warning, and handle nulls manually
    df_all = pd.read_csv(file_all)
    df_r3 = pd.read_csv(file_r3)
    df_r32 = pd.read_csv(file_r32)
    
    # Print initial sizes
    print(f"Loaded: All={len(df_all)}, R3={len(df_r3)}, R3.2={len(df_r32)}")

    # 1. Merge logic
    # Concatenate all
    combined_df = pd.concat([df_all, df_r3, df_r32], ignore_index=True)
    
    # 2. Fill missing Kd
    # Convert 'null' strings to NaN, coerce errors
    #combined_df['kd'] = pd.to_numeric(combined_df['kd'], errors='coerce')
    
    # Fill NaN with 1e-4
    #missing_before = combined_df['kd'].isna().sum()
    #combined_df['kd'] = combined_df['kd'].fillna(1e-4)
    #print(f"Filled {missing_before} missing Kd values with 1e-4")

    # 3. Deduplicate
    # "corrected using data from all_rounds" -> all_rounds data should be preserved.
    # Since we concatenated [all, r3, r32], the 'all' entries come first.
    # drop_duplicates(keep='first') will keep the 'all' entries and discard subsequent duplicates from r3/r32.
    unique_df = combined_df.drop_duplicates(subset='sequence', keep='first').copy()
    print(f"Merged {len(combined_df)} rows into {len(unique_df)} unique sequences.")

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
    out_path = os.path.join(args.base_dir, args.output_file)
    unique_df.to_csv(out_path, index=False)
    print(f"Saved merged & scored results to {out_path}")

    # Plotting & Correlations
    out_prefix = out_path.replace(".csv", "")
    
    plot_and_correlate(unique_df, 'kd', 'implicit_reward', 'Kd', 'Implicit Reward', out_prefix, color=colors[1])
    plot_and_correlate(unique_df, 'kd', 'ref_log_likelihood', 'Kd', 'Ref Log Likelihood', out_prefix, color=colors[1])
    plot_and_correlate(unique_df, 'kd', 'aligned_log_likelihood', 'Kd', 'Aligned Log Likelihood', out_prefix, color=colors[2])
    
    plot_binder_comparison(unique_df, out_prefix)
    
    plot_combined_likelihoods(unique_df, 'kd', 'ref_log_likelihood', 'aligned_log_likelihood', 
                              'Kd', 'Ref Log Likelihood', 'Aligned Log Likelihood', out_prefix)

def plot_combined_likelihoods(df, x_col, y_col1, y_col2, x_label, label1, label2, output_prefix):
    # Filter valid data
    data = df.dropna(subset=[x_col, y_col1, y_col2])
    if len(data) < 2:
        print(f"Not enough data to plot combined likelihoods")
        return

    plt.figure(figsize=(8, 6))
    
    # Plot 1: Ref (Light Gray)
    sns.regplot(data=data, x=x_col, y=y_col1, logx=True, 
                color='lightgray', 
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'lightgray'}
               )
    
    # Plot 2: Aligned (Orange)
    sns.regplot(data=data, x=x_col, y=y_col2, logx=True, 
                color='orange', 
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'orange'}
               )

    # Manually create legend handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=2.5, label=label1),
        Line2D([0], [0], color='orange', lw=2.5, label=label2)
    ]

    plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Log Likelihood')
    plt.title(f'Likelihood Comparison vs {x_label}')
    #plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_combined_log_likelihood_vs_kd.pdf", dpi=600)
    plt.close()
    print(f"Saved combined plot to {output_prefix}_combined_log_likelihood_vs_kd.pdf")

if __name__ == "__main__":
    main()
