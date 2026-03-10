
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set up visual style
sns.set_context("notebook")
sns.set_style("ticks")

# Config
CSV_COMP = "EGFR_test_all_rounds (Competition Buffer)_summary.csv"
CSV_ALL = "EGFR_test_all_rounds_summary.csv"

def get_base_name(name):
    return re.sub(r"_round.*", "", str(name))

def get_round(name):
    name = str(name)
    if name == "WT":
        return "WT"
    match = re.search(r"_round[_ ]?([a-zA-Z0-9.]+)$", name)
    if match:
        return match.group(1)
    return "Unspecified"

def load_and_clean_csv(path):
    df = pd.read_csv(path)
    
    # Extract metadata from 'name' column
    df["base_name"] = df["name"].apply(get_base_name)
    df["round"] = df["name"].apply(get_round)
    
    # Check data loss
    original_len = len(df)
    
    # Filter valid Kd
    df["kd"] = pd.to_numeric(df["kd"], errors='coerce')
    
    # Identify dropped: Only drop if KD is NaN. We keep "Unspecified" rounds.
    dropped_mask = df["kd"].isna()
    dropped_entries = df[dropped_mask]["name"].unique()
    
    # Clean
    clean = df[~dropped_mask]
    
    # Report
    print(f"\n--- Loading {path} ---")
    print(f"Original rows: {original_len}")
    print(f"Rows with valid Kd: {len(clean)}")
    loss_pct = (1 - len(clean)/original_len)*100
    print(f"Data loss (invalid Kd): {loss_pct:.1f}%")
    
    if len(dropped_entries) > 0:
        print(f"Dropped Entries (Invalid Kd) ({len(dropped_entries)} unique):")
        for ent in dropped_entries:
            print(f"  - {ent}")

    # Check for Unspecified rounds
    unspecified = df[df["round"] == "Unspecified"]["name"].unique()
    if len(unspecified) > 0:
        print(f"Unspecified Round Entries ({len(unspecified)} unique):")
        for ent in unspecified:
            print(f"  - {ent}")

    
    return clean

# 1. Load Data (CSV ONLY)
print("Processing CSVs...")
df_comp = load_and_clean_csv(CSV_COMP)
df_all = load_and_clean_csv(CSV_ALL)

# 2. Replicate Analysis
# Check variance across replicates
def analyze_replicates(df, label):
    # Group by sequence
    # Calc coefficient of variation (CV) for Kd
    # We keep the first 'base_name' and 'round' for display
    stats_df = df.groupby("sequence").agg({
        "kd": ["mean", "std", "count"],
        "base_name": "first",
        "round": "first"
    })
    
    # Flatten columns
    stats_df.columns = ["_".join(col).strip() for col in stats_df.columns.values]
    stats_df = stats_df.rename(columns={
        "kd_mean": "mean", 
        "kd_std": "std", 
        "kd_count": "count",
        "base_name_first": "base_name",
        "round_first": "round"
    })

    stats_df["cv"] = stats_df["std"] / stats_df["mean"]
    
    # Only meaningful for n > 1
    multi_rep = stats_df[stats_df["count"] > 1]
    
    high_cv = multi_rep[multi_rep["cv"] > 0.5] 
    
    print(f"\n[{label}] Replicate Analysis:")
    print(f"  Total unique sequences tested: {len(stats_df)}")
    print(f"  Sequences with >1 replicate: {len(multi_rep)}")
    print(f"  Sequences with CV > 0.5 (high variability): {len(high_cv)}")
    
    if len(high_cv) > 0:
        print("  Top 3 most variable mutants:")
        print(high_cv.sort_values("cv", ascending=False)[["base_name", "round", "mean", "cv"]].head(3))
        
    return multi_rep["cv"].mean()

avg_cv_comp = analyze_replicates(df_comp, "Competition Buffer")
avg_cv_all = analyze_replicates(df_all, "Standard Buffer")

# 3. Aggregate for Comparison (Geometric Mean)
def aggregate(df):
    # Geometric mean for Kd, keep representative name/round
    agg = df.groupby("sequence", as_index=False).agg({
        "kd": lambda x: np.exp(np.mean(np.log(x))),
        "base_name": "first",
        "round": "first"
    })
    return agg

agg_comp = aggregate(df_comp)
agg_all = aggregate(df_all)

# 4. Statistical Comparison: Buffer Effect
# Merge on SEQUENCE
merged = pd.merge(agg_comp, agg_all, on="sequence", suffixes=("_comp", "_std"))
# We can drop one set of name/round or keep both to check consistency, but for now we just use comp ones or std ones
# Let's clean up the merged df to have one Name column
merged["display_name"] = merged["base_name_comp"] # Use competition buffer name as primary

print(f"\n--- Statistical Comparison (Buffer Effect) ---")
print(f"Shared sequences between conditions: {len(merged)}")

# Paired Test
log_kd_comp = np.log(merged["kd_comp"])
log_kd_std = np.log(merged["kd_std"])

# T-test
t_stat, p_val_t = stats.ttest_rel(log_kd_comp, log_kd_std)
# Wilcoxon
w_stat, p_val_w = stats.wilcoxon(log_kd_comp, log_kd_std)

print(f"Paired T-test (log Kd): p = {p_val_t:.4e}")
print(f"Wilcoxon Signed-Rank: p = {p_val_w:.4e}")

mean_diff_log = np.mean(log_kd_comp - log_kd_std)
fold_change_avg = np.exp(mean_diff_log)
print(f"Mean log-diff (Comp - Std): {mean_diff_log:.3f}")
print(f"Average Fold Change (Kd Comp / Kd Std): {fold_change_avg:.2f}x")

# Pearson Correlation
r, p_corr = stats.pearsonr(log_kd_comp, log_kd_std)
print(f"Pearson Correlation (log Kd): r = {r:.3f} (p={p_corr:.4e})")

# 5. Advanced Analysis: Hamming Distance & Improvement
print("\n--- Advanced Analysis ---")

# Helper: Hamming Distance
def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        return max(len(s1), len(s2)) - min(len(s1), len(s2)) + sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def analyze_improvement(df, label, wt_seq="", wt_kd=None):
    print(f"\n[{label}] Generation Improvement (Round 1 -> Round 3.3):")
    
    # Identify WT if not provided or just for info
    # (WT might not be in every specific filtered df if we passed a subset, but here we pass full agg)
    
    r1 = df[df["round"] == "1"]
    r33 = df[df["round"] == "3.3"]
    
    if len(r1) > 0 and len(r33) > 0:
        avg_kd_r1 = np.exp(np.mean(np.log(r1["kd"])))
        avg_kd_r33 = np.exp(np.mean(np.log(r33["kd"])))
        improvement = avg_kd_r1 / avg_kd_r33
        
        print(f"  Avg Kd Round 1 ({len(r1)} seqs): {avg_kd_r1:.4e}")
        print(f"  Avg Kd Round 3.3 ({len(r33)} seqs): {avg_kd_r33:.4e}")
        print(f"  Improvement Factor: {improvement:.2f}x")
    else:
        print(f"  Insufficient data. R1: {len(r1)}, R3.3: {len(r33)}")
        
    # Top Performers (only for Standard usually, but let's do it if prompted or just keep Standard for detail)
    if label == "Standard Buffer" and len(r33) > 0 and wt_seq and wt_kd:
        top2 = r33.sort_values("kd", ascending=True).head(2)
        print(f"\n[{label}] Top 2 Performers in Round 3.3:")
        for idx, row in top2.iterrows():
            kd = row["kd"]
            seq = row["sequence"]
            name = row["base_name"]
            dist = hamming_distance(wt_seq, seq)
            fold_imp = wt_kd / kd
            
            print(f"  * {name}:")
            print(f"     Kd: {kd:.4e}")
            print(f"     Hamming Dist to WT: {dist}")
            print(f"     Fold Improvement vs WT: {fold_imp:.2f}x")

# Get WT details from Standard (most reliable usually)
wt_row = agg_all[agg_all["round"] == "WT"]
if len(wt_row) > 0:
    wt_kd_val = wt_row.iloc[0]["kd"]
    wt_seq_val = wt_row.iloc[0]["sequence"]
    print(f"WT Stats (Standard): Kd={wt_kd_val:.4e}")
else:
    wt_kd_val = 1.0; wt_seq_val = ""

# Run for both
analyze_improvement(agg_all, "Standard Buffer", wt_seq_val, wt_kd_val)
analyze_improvement(agg_comp, "Competition Buffer", wt_seq_val, wt_kd_val)

# 6. Visual Analysis Summary
plt.figure(figsize=(10, 5))

# Subplot 1: Correlation
plt.subplot(1, 2, 1)
sns.scatterplot(x=merged["kd_comp"], y=merged["kd_std"], alpha=0.6)
plt.xscale("log"); plt.yscale("log")
plt.plot([merged["kd_comp"].min(), merged["kd_comp"].max()], 
         [merged["kd_comp"].min(), merged["kd_comp"].max()], 
         '--', color='red', alpha=0.5, label="y=x")
plt.xlabel("Kd (Competition)")
plt.ylabel("Kd (Standard)")
plt.title(f"Buffer Correlation (r={r:.2f})")
plt.legend()

# Subplot 2: Residuals/Shift
plt.subplot(1, 2, 2)
# Fold change ratio
fold_changes_arr = merged["kd_comp"] / merged["kd_std"]
sns.histplot(np.log10(fold_changes_arr), kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("log10(Kd Comp / Kd Std)")
plt.title("Shift in Affinity (Distribution)")

plt.tight_layout()
plt.savefig("analysis_summary.png")
print("\nSaved analysis plots to 'analysis_summary.png'")
