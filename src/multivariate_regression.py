import statsmodels.api as sm


def run_multivariate(df, output_dir):
    features = [
        "DR","log_std","highlight_ratio",
        "shadow_ratio","skew","kurtosis"
    ]

    X = df[features]
    y = df["contrast_loss"]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    with open(f"{output_dir}/multivariate_summary.txt","w") as f:
        f.write(model.summary().as_text())

    return model.rsquared