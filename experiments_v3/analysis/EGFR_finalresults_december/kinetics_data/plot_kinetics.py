import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# Get all CSV files
data_dir = "/users/nferruz/fstocco/Desktop/EGFR_analysis/kinetics_data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
colors = ['#F8BA00', '#4F1AE1', '#f3f3f3']  # orange, purple, gray

plt.figure(figsize=(10, 6))  # default for any standalone figures
font = {'family': 'sans-serif', 'size': 30}
plt.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
plt.rcParams['axes.grid'] = False

# Read and combine all files
all_data = []
for f in csv_files:
    df = pd.read_csv(f)
    # Extract concentration from filename (last number before .csv)
    conc = os.path.basename(f).replace(".csv", "").split("_")[-1]
    df["concentration"] = float(conc)
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Plot
plt.figure(figsize=(10, 6))
palette = sns.light_palette("#4F1AE1", n_colors=4, reverse=False)  # shades from #4F1AE1
sns.lineplot(data=df_all, x="t", y="y", linewidth=8, hue="concentration", palette=palette, legend=False)

plt.xlabel("Time")
plt.ylabel("Signal (AU)")
#plt.title("Kinetics Data")
#plt.legend(title="Concentration")
plt.savefig(os.path.join(data_dir, "kinetics_plot.pdf"), dpi=600, bbox_inches="tight")
plt.close()

print("Done: kinetics_plot.pdf")
