import torch
import pandas as pd
from tqdm import tqdm


EV_SHIFTS = [-2, -1, 0, 1, 2]


def apply_exposure_shift(hdr, ev):
    scale = 2.0 ** ev
    return hdr * scale


def exposure_sensitivity_analysis(
    scenes,
    read_exr,
    to_tensor,
    OPERATORS,
    stability_function,
    output_csv
):
    results = []

    for scene in tqdm(scenes, desc="Exposure Robustness"):
        hdr_np = read_exr(scene)
        hdr_base = to_tensor(hdr_np)

        for op_name, operator in OPERATORS.items():

            stability_values = []

            for ev in EV_SHIFTS:
                hdr_shifted = apply_exposure_shift(hdr_base, ev)
                ldr = operator(hdr_shifted)

                stability = stability_function(hdr_shifted, ldr)
                stability_values.append(stability)

            esi = torch.tensor(stability_values).std().item()

            results.append({
                "scene": scene,
                "operator": op_name,
                "exposure_sensitivity_index": esi
            })

        del hdr_base
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    return df