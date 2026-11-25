"""Extrapolation MSE of the displacement channel at step 39 and at the last step,
per run and model: the median over the 30 extrapolation ICs (an IC whose MSE is
infinite or NaN counts as infinite). zero_u is the MSE of a zero prediction at the
last step, the mean square of the normalised target.

Usage, from the repository root:
    FA2023_OUTPUT=output uv run python docs/results/full-spectrum-sweep/final_mse.py \
        output/fullspectrumsweep final_mse.feather
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root

from evaluate_extrapolation import extrapolation_data  # noqa: E402
from platefno.util.conf import get_config  # noqa: E402
from platefno.util.eval import (  # noqa: E402
    calculate_mse_per_timestep,
    get_device,
    get_run_dirs,
    load_models_from_dir,
)

STEPS = 2000
NUM_VARIATIONS = 30


def main(exp_dir, out_file):
    rows = []
    for run_dir in get_run_dirs(exp_dir):
        cfg = get_config(run_dir)
        models = load_models_from_dir(run_dir)
        device = get_device(cfg)
        model_input, target = (
            t.to(device) for t in extrapolation_data(run_dir, cfg, STEPS, NUM_VARIATIONS)
        )
        zero_u = np.median((target[:, -1, ..., 0] ** 2).mean(dim=(1, 2)).cpu().numpy())
        all_mse = calculate_mse_per_timestep(models, model_input, target)
        for name, mse in zip(("gru", "rnn", "ref"), all_mse):
            mse = np.nan_to_num(mse, nan=np.inf)  # (IC, step, channel)
            rows.append(
                {
                    "run_dir": run_dir,
                    "model": name,
                    "step39_u": np.median(mse[:, 38, 0]),
                    "final_u": np.median(mse[:, -1, 0]),
                    "zero_u": zero_u,
                }
            )
        print(run_dir, flush=True)
    pd.DataFrame(rows).to_feather(out_file)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
