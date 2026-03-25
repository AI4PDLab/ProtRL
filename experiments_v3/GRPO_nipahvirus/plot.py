#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Process CSV files for normalization, scoring, and plotting.")
parser.add_argument("--d_path", type=str, required=True, help="Directory containing logs_output.csv")
args = parser.parse_args()

in_csv = os.path.join(args.d_path, "logs_output.csv")
out_png = os.path.join(args.d_path, "ProtWrap_Plot.png")

plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "sans-serif"
sns.set_theme(style="whitegrid", palette="coolwarm")

df = pd.read_csv(in_csv, on_bad_lines="skip")
df_norm = df.copy()

def divmax(s: pd.Series):
    a = s.astype(float).to_numpy()
    mx = np.nanmax(np.abs(a))
    mx = mx + (mx == 0)
    mx = np.where(np.isfinite(mx), mx, 1.0)
    return s.astype(float) / mx

df_norm["length_norm"] = (90 - df_norm["lenght"]).abs()
df_norm["b_plddt"] = df_norm["b_plddt"] / 0.90
df_norm["i_plddt"] = df_norm["i_plddt"] / 0.90
df_norm["i_ptm"]   = df_norm["i_ptm"]   / 0.50

for col in [
    "ipsae","b_pae","pMPNN","hydrophobicity","REU","interface_dSASA","contact_probs",
    "binder_sasa","delta_delta_interaction","helicity","lis","d0res","d0chn","d0dom",
    "iptm_d0chn","pdockq2","pdockq","ptm","t_pae","i_pae","d_rmsd","shape_complimentary"
    ]:
    if col in df_norm:
        df_norm[col] = divmax(df_norm[col])

df_norm["score"] = (
            #+ df_norm['i_plddt']
            - 0.1 * df_norm["length_norm"]
            #- 0.8 * df_norm['REU']
            #+ 0.8 * df_norm["b_plddt"]
            #+ 0.8 * df_norm["i_ptm"]
            - df_norm["b_pae"]
            - df_norm['i_pae']
            #- 0.5 * df_norm["pMPNN"]
            + 0.3 * df_norm["shape_complimentary"]
            #- 0.4 * df_norm["hydrophobicity"]
            + 0.7 * df_norm["interface_dSASA"]
            #- 0.7 * df_norm["uns_hydrogens"]
            - df_norm["d_rmsd"]
            #- 0.5 * df_norm["binder_sasa"]
            #+ 0.6 * df_norm["delta_delta_interaction"]
            #- 0.5 * df_norm['helicity']
            + df_norm['ipsae']
            #+ df_norm["iptm_af"]
            + df_norm["lis"]
            #+ df_norm["d0res"]
            #+ df_norm["d0dom"]
            + df_norm["iptm_d0chn"]
            #+ df_norm["pdockq2"]
            #+ df_norm["pdockq"]
            #+ df_norm["ptm"]
            - df_norm["t_pae"]
            #+ 1 * df_norm["numbers_clusters"]
            #- 10 * df_norm["seq_identity"]
        ) / 12

df["score"] = df_norm["score"]

filter_criteria = [
    ("ipsae", 0.80406, "higher"),
    ("i_pae", 7.321748129856925, "lower"),
    ("lis", 0.5297, "higher"),
    ("iptm_d0chn", 0.854868, "higher"),
    ("ptm", 0.88, "higher"),
    ("b_pae", 7.300776772247361, "lower"),
    ("t_pae", 9.977362190159084, "lower"),
    ("pdockq2", 0.5951, "higher"),
    ("pdockq", 0.6019, "higher"),
    ("d0dom", 8.07, "higher"),
    ("interface_dSASA", 3205.82, "higher")
]
filter_map = {k: {"value": v, "direction": d.lower()} for (k, v, d) in filter_criteria}

plot_cols = [
    "lenght","b_plddt","i_plddt","i_ptm","ipsae","b_pae","pMPNN","shape_complimentary",
    "hydrophobicity","REU","interface_dSASA","uns_hydrogens","score","d_rmsd","binder_sasa",
    "delta_delta_interaction","helicity","i_pae","lis","d0res","d0chn","d0dom","iptm_d0chn",
    "pdockq2","pdockq","ptm","t_pae", "seq_identity", "numbers_clusters"
]

for c in plot_cols:
    if c not in df.columns:
        df[c] = np.nan

if "iteration_num" not in df.columns:
    df["iteration_num"] = np.arange(len(df))
for c in plot_cols + ["iteration_num"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["iteration_num"] = pd.Categorical(df["iteration_num"])

num_plots = len(plot_cols)
ncols = 4
nrows = math.ceil(num_plots / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), sharex=False)
axes = axes.flatten()

for idx, col in enumerate(plot_cols):
    ax = axes[idx]
    plot_df = df[["iteration_num", col]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        ax.set_title(col)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1)
        continue
    sns.boxplot(data=plot_df, x="iteration_num", y=col, ax=ax, fliersize=0, showmeans=True,
                meanprops={"marker": "D", "markeredgecolor": "black"})
    #sns.stripplot(data=plot_df, x="iteration_num", y=col, ax=ax, size=3, jitter=True, alpha=0.6, color="black")
    ax.set_title(col)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    if col in filter_map:
        thr = filter_map[col]["value"]
        direction = filter_map[col]["direction"]
        color = "tab:green" if direction == "higher" else "tab:red"
        label = f"{'≥' if direction == 'higher' else '≤'} {thr:g}"
        ax.axhline(thr, linestyle="--", linewidth=1.5, color=color, label=label, zorder=1)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(min(y0, thr), max(y1, thr))
        try:
            x_right = ax.get_xticks()[-1] if len(ax.get_xticks()) else 0
            ax.text(x_right, thr, f" {label}", va="center", ha="left", fontsize=10, color=color)
        except Exception:
            pass
        ax.legend(loc="best", fontsize=10, frameon=True)

for j in range(num_plots, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(out_png, dpi=300)
print(f"Plotting complete. Image saved as '{out_png}'")
