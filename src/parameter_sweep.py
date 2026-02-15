import pandas as pd
import torch
from tqdm import tqdm


def parameter_sweep(
    hdr,
    scene_name,
    operator_class,
    param_grid,
    stability_function,
    device
):
    results = []

    for param_name, values in param_grid.items():
        for val in values:
            mapper = operator_class(param_name, **{param_name: val})
            ldr = mapper.apply(hdr)

            stability = stability_function(hdr, ldr)

            results.append({
                "scene": scene_name,
                "parameter": param_name,
                "value": val,
                "stability": stability
            })

            torch.cuda.empty_cache()

    return pd.DataFrame(results)