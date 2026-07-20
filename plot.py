import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import argparse
import os

def main():
    """Reads logs.csv and plots sequence length over iterations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to read logs and save plot")
    args = parser.parse_args()

    logs_path = os.path.join(args.output_dir, 'logs.csv')

    # Read the CSV
    try:
        df = pd.read_csv(logs_path)
    except FileNotFoundError:
        print(f"Error: logs.csv not found at {logs_path}.")
        # Don't exit with error, just return, so pipeline doesn't crash if logs aren't ready yet
        return

    # Plot length over iteration_num
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='iteration_num', y='length')
    plt.title('Sequence Length over Iterations')
    plt.xlabel('Iteration Number')
    plt.ylabel('Length')

    # Save the plot
    output_file = os.path.join(args.output_dir, 'length_over_iterations.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
