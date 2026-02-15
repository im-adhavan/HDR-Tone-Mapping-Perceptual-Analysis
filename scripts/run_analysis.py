import os
import pandas as pd
from tqdm import tqdm
import torch

from src.dataset_loader import read_exr, list_scenes
from src.gpu_utils import to_tensor, DEVICE
from src.tone_mapping import ToneMapper
from src.perceptual_metrics import clipping_ratio, contrast_loss, entropy
from src.scene_features import (
    dynamic_range,
    log_luminance_std,
    highlight_energy_ratio,
    shadow_mass_ratio,
    skewness,
    kurtosis
)
from src.operator_analysis import (
    compute_stability_index,
    operator_ranking,
    operator_significance,
    compute_exposure_stability_index
)
from src.multivariate_regression import run_multivariate
from src.advanced_analysis import (
    run_pca,
    cross_validated_regression,
    scene_clustering
)
from src.exposure_analysis import exposure_sensitivity_analysis
from src.optimization import optimize_dataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HDR_FOLDER = os.path.join(PROJECT_ROOT, "data", "hdr_exr")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Using device:", DEVICE)


OPERATORS = {
    "reinhard": lambda hdr: ToneMapper("reinhard", exposure=1.0).apply(hdr),
    "drago": lambda hdr: ToneMapper("drago", bias=0.85).apply(hdr),
    "filmic": lambda hdr: ToneMapper("filmic").apply(hdr),
    "gamma": lambda hdr: ToneMapper("gamma", gamma=2.2).apply(hdr),
    "reinhard_local": lambda hdr: ToneMapper("reinhard_local", sigma=3.0).apply(hdr),
}


scene_names = list_scenes(HDR_FOLDER)
rows = []

for scene in tqdm(scene_names, desc="Full GPU HDR Analysis"):

    hdr_np = read_exr(os.path.join(HDR_FOLDER, f"{scene}.exr"))
    hdr = to_tensor(hdr_np)

    DR = dynamic_range(hdr)
    log_std = log_luminance_std(hdr)
    highlight_ratio = highlight_energy_ratio(hdr)
    shadow_ratio = shadow_mass_ratio(hdr)
    skew = skewness(hdr)
    kurt = kurtosis(hdr)

    for op_name, operator in OPERATORS.items():
        ldr = operator(hdr)

        rows.append({
            "scene": scene,
            "operator": op_name,
            "contrast_loss": contrast_loss(hdr, ldr),
            "clipping": clipping_ratio(ldr),
            "entropy": entropy(ldr),
            "DR": DR,
            "log_std": log_std,
            "highlight_ratio": highlight_ratio,
            "shadow_ratio": shadow_ratio,
            "skew": skew,
            "kurtosis": kurt
        })

    del hdr
    torch.cuda.empty_cache()


df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "full_results.csv"), index=False)


df = compute_stability_index(df)

ranking = operator_ranking(df, RESULTS_DIR)
significance = operator_significance(df, RESULTS_DIR)


r2 = run_multivariate(df, RESULTS_DIR)


variance = run_pca(df, RESULTS_DIR)
cv_r2 = cross_validated_regression(df, RESULTS_DIR)
scene_clustering(df, RESULTS_DIR)


print("\nRunning Exposure Robustness Evaluation...")

esi_df = exposure_sensitivity_analysis(
    scenes=[os.path.join(HDR_FOLDER, f"{s}.exr") for s in scene_names],
    read_exr=read_exr,
    to_tensor=to_tensor,
    OPERATORS=OPERATORS,
    stability_function=compute_exposure_stability_index,
    output_csv=os.path.join(RESULTS_DIR, "exposure_sensitivity.csv")
)

operator_esi = (
    esi_df.groupby("operator")["exposure_sensitivity_index"]
    .mean()
    .sort_values()
)

operator_esi.to_csv(
    os.path.join(RESULTS_DIR, "operator_exposure_sensitivity_mean.csv")
)


print("\nRunning Optimal Parameter Search (Reinhard Exposure)...")

reinhard_exposure_values = [0.2, 0.5, 1.0, 1.5, 2.0]

optimal_df = optimize_dataset(
    scenes=[os.path.join(HDR_FOLDER, f"{s}.exr") for s in scene_names],
    read_exr=read_exr,
    to_tensor=to_tensor,
    operator_name="reinhard",
    param_name="exposure",
    values=reinhard_exposure_values,
    stability_function=compute_exposure_stability_index,
    output_csv=os.path.join(RESULTS_DIR, "optimal_reinhard_exposure.csv")
)

print("\nOptimal Parameter Search Complete.")


print("\n========== FINAL SUMMARY ==========")
print("Multivariate R²:", r2)
print("Cross-validated R²:", cv_r2)
print("PCA variance:", variance)
print("\nExposure Sensitivity Ranking:")
print(operator_esi)