import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# File path
csv_file = "EGFR_merged_scored.csv"

colors = ['#F8BA00', '#4F1AE1', '#E6E6E6']  # orange, purple, gray

plt.figure(figsize=(6, 6))  # default for any standalone figures
font = {'family': 'sans-serif', 'size': 30}
plt.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
plt.rcParams['axes.grid'] = False

# Extract round from name
def get_round(name):
    name = str(name)
    match = re.search(r"_round[_ ]?([a-zA-Z0-9.]+)$", name)
    return match.group(1) if match else None

def get_base_name(name):
    return re.sub(r"_round.*", "", str(name))

# Load and process data
df = pd.read_csv(csv_file)
df["round"] = df["name"].apply(get_round)
df["base_name"] = df["name"].apply(get_base_name)
df["kd"] = pd.to_numeric(df["kd"], errors='coerce')
df = df.dropna(subset=["kd", "round"])

# Geometric mean for duplicates
df = df.groupby(["round", "base_name"], as_index=False)["kd"].apply(lambda x: np.exp(np.mean(np.log(x))))

# Filter rounds 1 and 3.3
df_filtered = df[df["round"].isin(["1", "3.3"])].copy()

# Plot: Boxplot + Scatter
plt.figure(figsize=(8, 5))
rounds_order = ["1", "3.3"]
colors = {"1": "#FFA500", "3.3": "#4F1AE1"}  # orange, purple

sns.boxplot(data=df_filtered, x="round", y="kd", color="0.9", fliersize=0, order=rounds_order, width=0.5)
sns.stripplot(data=df_filtered, x="round", y="kd", hue="round", palette=colors, 
              jitter=0.15, size=12, alpha=0.7, order=rounds_order, legend=False)

# Get control Kd values from full data
df_full = pd.read_csv(csv_file)
df_full["kd"] = pd.to_numeric(df_full["kd"], errors='coerce')
ctrl_01 = df_full[df_full["name"].str.contains("01_round_ctrl", na=False)]["kd"].mean()
wt = df_full[df_full["name"] == "WT"]["kd"].mean()

# Add dashed lines for controls
plt.axhline(y=ctrl_01, color="0.7", linestyle="--", linewidth=1.5, label="01 ctrl")
plt.axhline(y=wt, color="0.7", linestyle="--", linewidth=1.5, label="WT")
#plt.legend(loc="upper right", fontsize=10)

plt.yscale("log")
plt.ylabel("Kd")
plt.xlabel("Round")
plt.savefig("plot_round1_vs_3.3.pdf", dpi=600, bbox_inches="tight")
plt.close()

print("Done: plot_round1_vs_3.3.pdf")
