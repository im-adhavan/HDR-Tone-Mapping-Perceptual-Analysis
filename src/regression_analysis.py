import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


def run_regression(df, output_dir):
    corr = df.corr(numeric_only=True)
    corr.to_csv(f"{output_dir}/correlation_matrix.csv")

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()

    slope, intercept, r_value, _, _ = linregress(df["DR"], df["contrast_loss"])
    r2 = r_value**2

    return slope, intercept, r2