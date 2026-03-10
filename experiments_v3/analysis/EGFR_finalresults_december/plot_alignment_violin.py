import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV (no header)
df = pd.read_csv("alignment_analysis/RPXEM4SM014-Alignment-HitTable.csv", header=None)
df.columns = ["name", "hit", "identity"] + [f"col{i}" for i in range(3, len(df.columns))]

# Get max identity per sequence
df_max = df.groupby("name")["identity"].max().reset_index()

# Extract round
def get_round(name):
    if "round_3.3" in str(name) or "_round3.3" in str(name): return "3.3"
    if "round_1" in str(name) or "_round1" in str(name): return "1"
    return None

df_max["round"] = df_max["name"].apply(get_round)
df_max = df_max.dropna(subset=["round"])

# Plot violin with orange/purple
colors = {"1": "#FFA500", "3.3": "#4F1AE1"}  # orange, purple
plt.figure(figsize=(10, 6))  # default for any standalone figures
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 14})

sns.violinplot(data=df_max, x="round", y="identity", order=["1", "3.3"], 
               hue="round", palette=colors, inner=None, legend=False, linewidth=0)
# Lighter shades for boxplots
box_colors = {"1": "#FFD699", "3.3": "#A88BE0"}  # light orange, light purple
sns.boxplot(data=df_max, x="round", y="identity", order=["1", "3.3"],
            hue="round", palette=box_colors, width=0.08, fliersize=0, legend=False)

plt.ylabel("Max Identity (%)")
plt.xlabel("Round")
plt.tight_layout()
plt.savefig("plot_alignment_violin.pdf", dpi=600, bbox_inches="tight")
plt.savefig("plot_alignment_violin.png", dpi=300, bbox_inches="tight")
plt.close()
print("Done: plot_alignment_violin.pdf")
