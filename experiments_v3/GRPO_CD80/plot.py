import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os 

# -----------------------------------------------------------------------------
# 1) styling
# -----------------------------------------------------------------------------
plt.rcParams['font.size']   = 20
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid", palette="coolwarm")

# -----------------------------------------------------------------------------
# 2) load & normalize
# -----------------------------------------------------------------------------
# 1. Point this to the folder containing your CSVs
csv_folder = '/home/woody/b114cb/b114cb23/ProtWrap_2'

csv_paths = glob.glob(os.path.join(csv_folder, '*.csv'))
df_list = [pd.read_csv(p) for p in csv_paths]
df = pd.concat(df_list, ignore_index=True)

# ensure iteration_num and all other features are numeric
df["iteration_num"] = pd.to_numeric(df["iteration_num"], errors="coerce")
for col in df.columns:
    if col not in ("name", "sequence", "iteration_num"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

# length_norm
df['length_norm'] = np.exp(
    -(((90.0 / df['lenght']) - 1.0) ** 2) / (0.5 ** 2)
)

# features to max-normalize
norm_feats = [
    'plddt', 'i_pTM', 'pAE_b', 'pAE_bt', 'pMPNN',
    'shape_complimentary', 'hydrophobicity', 'binder_score',
    'interface_dSASA', 'uns_hydrogens', 'contact_probs',
    'd_rmsd', 'binder_sasa', 'delta_delta_interaction'
]

# compute your weighted score
df["score"] = (
        df["length_norm"]
        * (
            0.80 * df["plddt"]
            + df["i_pTM"]
            - 0.8 * df["pAE_b"]
            - 0.8 * df["pAE_bt"]
            - 0.5 * df["pMPNN"]
            + 0.3 * df["shape_complimentary"]
            - 0.4 * df["hydrophobicity"]
            - 0.9 * df["binder_score"]
            - 0.3 * df["interface_dSASA"]
            - 0.7 * df["uns_hydrogens"]
            + 0.5 * df["contact_probs"]
            - 0.3 * df["d_rmsd"]
            - 0.5 * df["binder_sasa"]
            + 0.6* df["delta_delta_interaction"]
            - 0.5 * df['helicity']
                )
        )/15

# -----------------------------------------------------------------------------
# 3) plotting
# -----------------------------------------------------------------------------
plot_cols = ['length_norm'] + norm_feats + ['score']
num_plots  = len(plot_cols)
ncols      = 4
nrows      = math.ceil(num_plots / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 5 * nrows),
    sharex=False
)
axes = axes.flatten()

for idx, col in enumerate(plot_cols):
    ax = axes[idx]
    sns.boxplot(
        data=df,
        x='iteration_num',
        y=col,
        ax=ax,
        fliersize=0,
        showmeans=True,
        meanprops={'marker': 'D', 'markeredgecolor': 'black'}
    )
    sns.stripplot(
        data=df,
        x='iteration_num',
        y=col,
        ax=ax,
        size=3,
        jitter=True,
        alpha=0.6,
        color='black'
    )
    ax.set_title(col)
    #ax.set_ylim(0, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")

# turn off unused subplots
for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("ProtWrap_2.png")
