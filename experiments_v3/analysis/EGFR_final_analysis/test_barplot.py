import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Mock data
merged_df = pd.DataFrame({
    "dataset": ["A", "A", "B", "B"],
    "kd": [0.01, np.nan, 0.005, 0.0001]
})

df = merged_df.copy()
df["kd"] = df["kd"].fillna(1e-3)                 # NA -> 10^-3
df["entry"] = df.groupby("dataset").cumcount()

g = sns.FacetGrid(df, col="dataset", col_wrap=3, sharex=False, sharey=False, height=4)
g.map_dataframe(sns.barplot, x="entry", y="kd", estimator=None, errorbar=None)

for ax in g.axes.flat:
    # ax.set_yscale("log")
    ax.set_xticks([])                            # hides crowded x ticks (remove if you want them)

g.set_axis_labels("Entry", "KD (log)")
# g.set_titles("{col_name}")
plt.tight_layout()
# plt.savefig("test.png")
