import torch
import pandas as pd
from tqdm import tqdm
from .tone_mapping import ToneMapper


def find_optimal_parameter(
    hdr,
    operator_name,
    param_name,
    values,
    stability_function
):
    """
    For a single scene and operator,
    finds parameter value that minimizes stability score.
    """

    best_val = None
    best_score = float("inf")

    for val in values:
        mapper = ToneMapper(operator_name, **{param_name: val})
        ldr = mapper.apply(hdr)

        score = stability_function(hdr, ldr)

        if score < best_score:
            best_score = score
            best_val = val

        torch.cuda.empty_cache()

    return best_val, best_score


def optimize_dataset(
    scenes,
    read_exr,
    to_tensor,
    operator_name,
    param_name,
    values,
    stability_function,
    output_csv
):
    """
    Runs optimal parameter search across entire dataset.
    """

    results = []

    for scene_path in tqdm(scenes, desc="Optimal Parameter Search"):
        scene_name = scene_path.split("\\")[-1].replace(".exr", "")

        hdr_np = read_exr(scene_path)
        hdr = to_tensor(hdr_np)

        best_val, best_score = find_optimal_parameter(
            hdr,
            operator_name,
            param_name,
            values,
            stability_function
        )

        results.append({
            "scene": scene_name,
            "operator": operator_name,
            "optimal_value": best_val,
            "optimal_stability": best_score
        })

        del hdr
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    return df