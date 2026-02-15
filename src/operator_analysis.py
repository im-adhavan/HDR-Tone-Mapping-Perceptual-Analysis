import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import torch
from .perceptual_metrics import clipping_ratio, contrast_loss, entropy


def compute_stability_index(df):
    """
    Stability index combines:
    - absolute contrast loss
    - clipping ratio
    - entropy

    Higher = more unstable
    """
    df["stability"] = (
        df["contrast_loss"].abs()
        + df["clipping"]
        + df["entropy"]
    )
    return df


def operator_ranking(df, output_dir):
    ranking = df.groupby("operator")["stability"].mean().sort_values()
    ranking.to_csv(f"{output_dir}/operator_ranking.csv")

    plt.figure(figsize=(8,6))
    ranking.plot(kind="bar")
    plt.ylabel("Mean Stability Index")
    plt.title("Operator Stability Ranking")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/operator_ranking.png")
    plt.close()

    return ranking


def operator_significance(df, output_dir):
    operators = df["operator"].unique()
    results = []

    for i in range(len(operators)):
        for j in range(i + 1, len(operators)):
            op1 = operators[i]
            op2 = operators[j]

            d1 = df[df["operator"] == op1]["stability"]
            d2 = df[df["operator"] == op2]["stability"]

            stat, p = ttest_rel(d1.values, d2.values)
            results.append([op1, op2, p])

    sig_df = pd.DataFrame(
        results,
        columns=["Operator1", "Operator2", "p_value"]
    )

    sig_df.to_csv(f"{output_dir}/operator_significance.csv", index=False)
    return sig_df


def compute_exposure_stability_index(hdr, ldr):
    """
    Computes stability metric for one HDR/LDR pair.
    Used inside exposure robustness loop.
    """
    return (
        abs(contrast_loss(hdr, ldr))
        + clipping_ratio(ldr)
        + entropy(ldr)
    )