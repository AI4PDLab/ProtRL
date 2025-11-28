import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing benchmark results")
    args = parser.parse_args()

    # Find all run directories (subdirectories in results_dir)
    # Assuming structure: results/<timestamp>/<run_id>/logs.csv
    # But run_benchmark.sh creates results/<timestamp>/<run_id>
    # So if we run this script, we might need to point it to specific timestamp folder or it finds all?
    # Let's assume the user runs it pointing to a specific timestamp folder, e.g., results/2023...
    # Or we can just look recursively.
    
    print(f"Scanning {args.results_dir} for logs...")
    
    all_data = []
    
    # Walk through the directory to find logs.csv
    for root, dirs, files in os.walk(args.results_dir):
        if "logs.csv" in files:
            logs_path = os.path.join(root, "logs.csv")
            run_id = os.path.basename(root) # The folder name is the run_id
            
            try:
                df = pd.read_csv(logs_path)
                df["run_id"] = run_id
                
                # Parse run_id to extract parameters for better labeling if possible
                # run_id format: method_betaX_lrY_isZ
                # We can keep run_id as is for now, or split it.
                
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {logs_path}: {e}")

    if not all_data:
        print("No logs found.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Extract method from run_id
    # run_id format: method_betaX_lrY_isZ
    # Methods: pLM_GRPO, weighted_DPO, trl_GRPO
    def get_method(run_id):
        if run_id.startswith("pLM_GRPO"):
            return "pLM_GRPO"
        elif run_id.startswith("weighted_DPO"):
            return "weighted_DPO"
        elif run_id.startswith("trl_GRPO"):
            return "trl_GRPO"
        else:
            return "other"

    combined_df["method"] = combined_df["run_id"].apply(get_method)
    
    # Save combined logs
    combined_csv = os.path.join(args.results_dir, "benchmark_combined_logs.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"Combined logs saved to {combined_csv}")

    # Plotting
    # Create a FacetGrid with one plot per method
    # Share x axis, but maybe not y axis if scales differ significantly? User didn't specify.
    # Usually better to share y for comparison, but if one explodes it hides others.
    # Let's share both for now to allow direct comparison.
    
    g = sns.FacetGrid(combined_df, col="method", col_wrap=3, height=5, aspect=1.5, sharey=False)
    g.map_dataframe(sns.lineplot, x="iteration_num", y="length", hue="run_id", marker="o")
    g.add_legend()
    g.set_titles("{col_name}")
    g.set_axis_labels("Iteration", "Sequence Length")
    
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Sequence Length over Iterations by Method")
    
    output_plot = os.path.join(args.results_dir, "benchmark_comparison.png")
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    main()
