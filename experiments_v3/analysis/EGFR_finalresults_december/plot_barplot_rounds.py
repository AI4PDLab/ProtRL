import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("EGFR_merged_scored.csv")
df["kd"] = pd.to_numeric(df["kd"], errors='coerce')

# Categorize
def get_category(name):
    if "ctrl" in str(name).lower(): return "ctrl"
    if "round_3.3" in str(name) or "_round3.3" in str(name): return "3.3"
    if "round_1" in str(name) or "_round1" in str(name): return "1"
    return None

df["category"] = df["name"].apply(get_category)
df = df.dropna(subset=["kd", "category"])

# Geometric mean for duplicates
df = df.groupby(["name", "category"], as_index=False)["kd"].apply(lambda x: np.exp(np.mean(np.log(x))))
df = df.sort_values("kd", ascending=True)

# WT line
wt_kd = pd.read_csv("EGFR_merged_scored.csv")
wt_kd = pd.to_numeric(wt_kd[wt_kd["name"] == "WT"]["kd"], errors='coerce').mean()

# Colors: Round 1 = orange, Round 3.3 = purple, ctrl = gray
colors = {"1": "#FFA500", "3.3": "#4F1AE1", "ctrl": "#E6E6E6"}
bar_colors = [colors[c] for c in df["category"]]

# Plot
plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})

x = range(len(df))
plt.bar(x, df["kd"], color=bar_colors, edgecolor="none")
plt.axhline(y=wt_kd, color="orange", linestyle="--", linewidth=2)

plt.yscale("log")
plt.ylabel("Kd (Log Scale)")
plt.xlabel("Sequence")
plt.xticks(x, df["name"], rotation=90, fontsize=8)
plt.title("Kd Comparison: Round 1 vs 3.3 vs Controls")

plt.tight_layout()

plt.savefig("plot_barplot_rounds.pdf", dpi=600, bbox_inches="tight")
plt.savefig("plot_barplot_rounds.png", dpi=300, bbox_inches="tight")
plt.close()
print("Done: plot_barplot_rounds.pdf")
