import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np


def run_pca(df, output_dir):
    features = ["DR","log_std","highlight_ratio","shadow_ratio","skew","kurtosis"]
    X = df[features].drop_duplicates()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7,6))
    plt.scatter(components[:,0], components[:,1])
    plt.title("PCA of Scene Radiometric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"pca_plot.png"))
    plt.close()

    return pca.explained_variance_ratio_


def cross_validated_regression(df, output_dir):
    features = ["DR","log_std","highlight_ratio","shadow_ratio","skew","kurtosis"]
    X = df[features].values
    y = df["contrast_loss"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()

    r2_scores = []
    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        r2_scores.append(model.score(X[test_idx], y[test_idx]))

    mean_r2 = np.mean(r2_scores)

    with open(os.path.join(output_dir,"cross_validated_r2.txt"),"w") as f:
        f.write(f"Mean CV R²: {mean_r2}\n")

    return mean_r2


def scene_clustering(df, output_dir):
    features = ["DR","log_std","highlight_ratio","shadow_ratio","skew","kurtosis"]
    X = df[features].drop_duplicates()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(7,6))
    plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters)
    plt.title("Scene Clustering")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"scene_clusters.png"))
    plt.close()