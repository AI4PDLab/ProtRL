import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    logs_path = os.path.join(args.output_dir, 'logs.csv')
    if not os.path.exists(logs_path):
        return

    try:
        df = pd.read_csv(logs_path)
    except Exception:
        return

    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'iteration_num' in numerical_cols:
        numerical_cols.remove('iteration_num')

    if not numerical_cols:
        return

    num_plots = len(numerical_cols)
    cols = 2
    rows = (num_plots + 1) // 2
    plt.figure(figsize=(10 * cols, 6 * rows))

    for i, col in enumerate(numerical_cols):
        plt.subplot(rows, cols, i + 1)
        sns.lineplot(data=df, x='iteration_num', y=col, marker='o')
        plt.title(col)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_plot.png'))

if __name__ == "__main__":
    main()
