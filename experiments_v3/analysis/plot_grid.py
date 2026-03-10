import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Style configuration
plt.figure(figsize=(10, 6))
font = {'family': 'sans-serif', 'size': 30}
plt.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2.5
# Define colors
colors = ['#E7B471', '#4F1AE1', '#7EC2C5']
sns.set_palette(sns.color_palette(colors))

def plot_grid(csv_paths):
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            num_cols = df.select_dtypes(include=['number']).columns
            
            # Determine x-axis
            x_col = None
            if 'iteration_num' in df.columns:
                x_col = 'iteration_num'
                # Exclude iteration_num from y-axis plots
                y_cols = [c for c in num_cols if c != 'iteration_num']
            else:
                y_cols = list(num_cols)
            
            n = len(y_cols)
            if n == 0:
                print(f"No variables to plot in {path}")
                continue

            # Single column layout
            cols_grid = 1
            rows_grid = n
            
            # Adjust height: 6 inches per row, width 8 inches for more squared look
            fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(8, 6*n), constrained_layout=True)
            
            # Ensure axes is iterable
            if n == 1:
                axes = [axes]

            for i, col in enumerate(y_cols):
                ax = axes[i]
                # Coherent colors: cycle through palette based on index
                c = colors[i % len(colors)]
                
                if x_col:
                    sns.lineplot(data=df, x=x_col, y=col, ax=ax, color=c)
                    if i < n - 1:
                        ax.set_xlabel('')
                        ax.set_xticklabels([]) # Hide x-axis numbers
                    else:
                        ax.set_xlabel(x_col)
                else:
                    sns.lineplot(data=df[col], ax=ax, color=c)
                
                ax.set_title(col)
                if 'tm_score' in col:
                    ax.set_ylim(0, 1)
                if 'clusters' in col:
                    ax.set_ylim(bottom=0)
                
                
                #ax.set_title(col)
                # Remove x-label for all but bottom plot to save space? 
               
            
            out_pdf = os.path.splitext(path)[0] + ".pdf"
            plt.savefig(out_pdf, dpi=600)
            plt.close(fig)
            print(f"Saved {out_pdf}")
            
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_grid.py <csv_file1> <csv_file2> ...")
        sys.exit(1)
    
    plot_grid(sys.argv[1:])
