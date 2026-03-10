
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# File paths
# File paths
csv_comp = "EGFR_test_all_rounds (Competition Buffer)_summary.csv"
csv_all = "EGFR_merged_scored.csv"

# Regex for extracting round
def get_round(name):
    # Ensure string
    name = str(name)
    if name == "WT":
        return "WT"
    match = re.search(r"_round[_ ]?([a-zA-Z0-9.]+)$", name)
    if match:
        return match.group(1)
    return "Unspecified"

def get_base_name(name):
    name = str(name)
    return re.sub(r"_round.*", "", name)

# 2. Process Data Function
def process_data(csv_file):
    df = pd.read_csv(csv_file)
    
    # Extract round and base_name directly from CSV 'name' column
    df["round"] = df["name"].apply(get_round)
    df["base_name"] = df["name"].apply(get_base_name)
    
    # 'kd' is sometimes 'null' string in CSV, force numeric
    df["kd"] = pd.to_numeric(df["kd"], errors='coerce')
    
    # Dropna for essential columns: kd
    # We DO NOT drop if round is "Unspecified" anymore, as long as KD is valid
    d = df.dropna(subset=["kd"]).copy()
    
    # Geometric mean for duplicates
    # Group by round and base_name, then average Kd
    d_grouped = d.groupby(["round", "base_name"], as_index=False)["kd"].apply(lambda x: np.exp(np.mean(np.log(x))))
    return d_grouped

# Process both datasets
print("Processing Competition Buffer data...")
df_comp_processed = process_data(csv_comp)

print("Processing All Rounds data...")
df_all_processed = process_data(csv_all)

# Helper function for plotting
def plot_rounds(data, title, filename):
    plt.figure(figsize=(6, 6))
    
    # Sort rounds
    rounds = sorted(data["round"].unique().astype(str))
    
    sns.boxplot(data=data, x="round", y="kd", color="0.85", fliersize=0, order=rounds)
    sns.stripplot(data=data, x="round", y="kd", jitter=0.15, size=12, alpha=0.6, order=rounds)
    
    plt.yscale("log")
    
    plt.title(title)
    plt.ylabel("Kd")
    plt.xlabel("Round")
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()

# 3. Generate Plots

# Plot 1: Competition Buffer
print("Generating Competition Buffer plot...")
plot_rounds(df_comp_processed, "Competition Buffer", "plot_competition_buffer.pdf")

# Plot 2: All Rounds
print("Generating All Rounds plot...")
plot_rounds(df_all_processed, "All Rounds Summary", "plot_all_rounds.pdf")

# Plot 3: Comparison
print("Generating Comparison plot...")
merged = pd.merge(df_comp_processed, df_all_processed, on="base_name", suffixes=("_comp", "_all"))

plt.figure(figsize=(8, 8)) # Increased size to accommodate labels
sns.scatterplot(data=merged, x="kd_comp", y="kd_all", s=50, alpha=0.7)

# Add Labels
for i, row in merged.iterrows():
    plt.text(row["kd_comp"], row["kd_all"], row["base_name"], 
             fontsize=6, alpha=0.7, ha='left', va='bottom')

# Log scale
plt.xscale("log")
plt.yscale("log")

# Add diagonal line
min_val = min(merged["kd_comp"].min(), merged["kd_all"].min())
max_val = max(merged["kd_comp"].max(), merged["kd_all"].max())
plt.plot([min_val, max_val], [min_val, max_val], ls="--", c="gray", alpha=0.5)

plt.xlabel("Kd - Competition Buffer")
plt.ylabel("Kd - All Rounds")
plt.title("Comparison")

plt.savefig("plot_comparison.pdf", dpi=600, bbox_inches="tight")
plt.close()

# Plot 4: Barplot Comparison (Round 1 vs 3.3 vs Controls)
def plot_bar_comparison(data, outfile):
    print(f"Generating Barplot Comparison ({outfile})...")
    
    # Identify WT / Controls: round="WT" or "ctrl" in name
    def is_target(row):
        r = str(row["round"])
        # Check exact matches
        if r in ["1", "3.3", "WT"]:
            return True
        # Check for control
        if "ctrl" in r.lower():
            return True
        return False
        
    subset = data[data.apply(is_target, axis=1)].copy()
    
    if subset.empty:
        print("Warning: No data for Rounds 1, 3.3 or WT.")
        return
        
    # Sort by Kd descending
    subset = subset.sort_values(by="kd", ascending=False).reset_index(drop=True)
    
    # WT Kd for line
    wt_rows = subset[subset["round"] == "WT"]
    wt_val = None
    if not wt_rows.empty:
        wt_val = wt_rows["kd"].mean()
        print(f"WT Kd for line: {wt_val}")
    
    plt.figure(figsize=(max(6, len(subset)*0.25), 6))
    
    # Palette
    def get_color(r):
        r_str = str(r)
        if r_str == "1": return "silver"
        if r_str == "3.3": return "#4F1AE1"
        if r_str == "WT" or "ctrl" in r_str.lower(): return "#F8BA00"
        return "gray"
        
    unique_rounds = subset["round"].unique()
    palette = {r: get_color(r) for r in unique_rounds}
    
    # Plot
    ax = sns.barplot(
        data=subset, 
        x=subset.index, 
        y="kd", 
        hue="round", 
        palette=palette, 
        dodge=False, 
        saturation=1
    )
    
    # Labels
    ax.set_xticklabels(subset["base_name"], rotation=90, fontsize=8)
    
    # WT Line
    if wt_val is not None:
        plt.axhline(y=wt_val, color="#F8BA00", linestyle="--", linewidth=1.5, label="WT Kd")
        
    plt.yscale("log")
    plt.ylabel("Kd (Log Scale)")
    plt.xlabel("Sequence")
    plt.title("Kd Comparison: Round 1 vs 3.3 vs Controls")
    plt.legend(title="Round")
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()

plot_bar_comparison(df_all_processed, "plot_barplot_1_vs_3.3_controls.pdf")

print("Done.")
