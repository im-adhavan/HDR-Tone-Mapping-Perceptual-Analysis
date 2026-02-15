import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def predict_optimal_exposure(full_results_csv,
                             optimal_csv,
                             output_dir):

    df_full = pd.read_csv(full_results_csv)
    df_opt = pd.read_csv(optimal_csv)

    scene_features = df_full.drop_duplicates("scene")[
        [
            "scene",
            "DR",
            "log_std",
            "highlight_ratio",
            "shadow_ratio",
            "skew",
            "kurtosis"
        ]
    ]

    df = pd.merge(scene_features, df_opt, on="scene")

    features = [
        "DR",
        "log_std",
        "highlight_ratio",
        "shadow_ratio",
        "skew",
        "kurtosis"
    ]

    X = df[features].values
    y = df["optimal_value"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    r2 = model.score(X_scaled, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, test_idx in kf.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        cv_scores.append(model.score(X_scaled[test_idx], y[test_idx]))

    cv_r2 = np.mean(cv_scores)

    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_
    })

    coef_df.to_csv(os.path.join(output_dir, "optimal_exposure_coefficients.csv"),
                   index=False)

    preds = model.predict(X_scaled)

    plt.figure(figsize=(6,6))
    plt.scatter(y, preds)
    plt.xlabel("True Optimal Exposure")
    plt.ylabel("Predicted Optimal Exposure")
    plt.title("Adaptive Tone Mapping Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             "optimal_exposure_prediction.png"))
    plt.close()

    with open(os.path.join(output_dir,
                           "optimal_exposure_prediction_summary.txt"),
              "w") as f:
        f.write(f"R² (train): {r2}\n")
        f.write(f"R² (cross-val): {cv_r2}\n")

    return r2, cv_r2