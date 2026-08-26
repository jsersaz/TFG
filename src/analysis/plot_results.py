import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(SRC_DIR)

from utils.utils import get_project_root


# ----------------------------------------
# LOAD RESULTS
# ----------------------------------------

def load_results(csv_filename):

    project_root = get_project_root()
    results_dir = os.path.join(project_root, "results")

    filepath = os.path.join(results_dir, csv_filename)

    df = pd.read_csv(filepath)

    return df


# ----------------------------------------
# FIGURE DIRECTORY
# ----------------------------------------

def get_figures_dir():

    project_root = get_project_root()
    figures_dir = os.path.join(project_root, "figures")

    os.makedirs(figures_dir, exist_ok=True)

    return figures_dir


# ----------------------------------------
# ACCURACY vs MODEL SIZE
# ----------------------------------------

def plot_accuracy_vs_size(df):

    plt.figure()

    plt.scatter(df["Size_KB"], df["Accuracy_mean"])

    for i, row in df.iterrows():
        plt.text(row["Size_KB"], row["Accuracy_mean"], row["Model"])

    plt.xlabel("Model size (KB)")
    plt.ylabel("Accuracy")

    plt.title("Accuracy vs Model Size")

    figures_dir = get_figures_dir()

    plt.savefig(os.path.join(figures_dir, "accuracy_vs_size.png"), dpi=300)

    plt.close()


# ----------------------------------------
# ACCURACY vs INFERENCE TIME
# ----------------------------------------

def plot_accuracy_vs_latency(df):

    plt.figure()

    plt.scatter(df["Inference_time_s"], df["Accuracy_mean"])

    for i, row in df.iterrows():
        plt.text(row["Inference_time_s"], row["Accuracy_mean"], row["Model"])

    plt.xlabel("Inference time (s)")
    plt.ylabel("Accuracy")

    plt.title("Accuracy vs Inference Time")

    figures_dir = get_figures_dir()

    plt.savefig(os.path.join(figures_dir, "accuracy_vs_latency.png"), dpi=300)

    plt.close()


# ----------------------------------------
# ACCURACY vs MACs
# ----------------------------------------

def plot_accuracy_vs_macs(df):

    if "MACs" not in df.columns:
        return

    plt.figure()

    plt.scatter(df["MACs"], df["Accuracy_mean"])

    for i, row in df.iterrows():
        plt.text(row["MACs"], row["Accuracy_mean"], row["Model"])

    plt.xlabel("MAC operations")
    plt.ylabel("Accuracy")

    plt.title("Accuracy vs Computational Cost")

    figures_dir = get_figures_dir()

    plt.savefig(os.path.join(figures_dir, "accuracy_vs_macs.png"), dpi=300)

    plt.close()


# ----------------------------------------
# PARETO FRONTIER
# ----------------------------------------

def plot_pareto_size_accuracy(df):

    df_sorted = df.sort_values("Size_KB")

    pareto = []

    best_acc = 0

    for _, row in df_sorted.iterrows():

        if row["Accuracy_mean"] > best_acc:
            pareto.append(row)
            best_acc = row["Accuracy_mean"]

    pareto_df = pd.DataFrame(pareto)

    plt.figure()

    plt.scatter(df["Size_KB"], df["Accuracy_mean"], alpha=0.4)

    plt.plot(pareto_df["Size_KB"], pareto_df["Accuracy_mean"], marker="o")

    for i, row in df.iterrows():
        plt.text(row["Size_KB"], row["Accuracy_mean"], row["Model"])

    plt.xlabel("Model size (KB)")
    plt.ylabel("Accuracy")

    plt.title("Pareto Frontier: Accuracy vs Model Size")

    figures_dir = get_figures_dir()

    plt.savefig(os.path.join(figures_dir, "pareto_accuracy_size.png"), dpi=300)

    plt.close()


# ----------------------------------------
# MAIN
# ----------------------------------------

def generate_all_figures(csv_filename):

    df = load_results(csv_filename)

    plot_accuracy_vs_size(df)

    plot_accuracy_vs_latency(df)

    plot_accuracy_vs_macs(df)

    plot_pareto_size_accuracy(df)

    print("\nFigures generated in /figures\n")


if __name__ == "__main__":

    generate_all_figures("har_results_multiseed.csv")
