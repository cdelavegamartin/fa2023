import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import os
import pathlib
import pandas as pd


from platefno.util.conf import get_config
from platefno.util.eval import (
    get_run_dirs,
    calculate_mse_per_timestep,
    load_models_from_dir,
    load_data,
    calculate_mse_crossval,
)
from platefno.util.plot import plot_mse_per_timestep


def evaluate_run(dir_name):
    model_gru, model_rnn, model_ref = load_models_from_dir(dir_name)
    cfg = get_config(dir_name)

    # Set torch device if posible if not cpu
    if torch.cuda.is_available() and cfg.train.device == "cuda":
        device = torch.device(cfg.train.device)
    else:
        device = torch.device("cpu")

    validation_input, validation_output = load_data(dir_name)
    validation_input = validation_input.to(device)
    validation_output = validation_output.to(device)

    # Calculate MSE per step
    (
        val_gru_mse_per_step,
        val_rnn_mse_per_step,
        val_ref_mse_per_step,
    ) = calculate_mse_per_timestep(
        (model_gru, model_rnn, model_ref),
        validation_input,
        validation_output,
    )

    plot_mse_per_timestep(
        (
            val_gru_mse_per_step,
            val_rnn_mse_per_step,
            val_ref_mse_per_step,
        ),
        dir_name,
    )
    calculate_mse_crossval(dir_name)

    return


if __name__ == "__main__":
    import sys
    import time

    # get the directory name from the command line
    dir_name = sys.argv[1]
    # evaluate_run(dir_name)
    # time how long it takes
    timer_start = time.time()
    # print(len(get_run_dirs(dir_name)))

    # loop over all the runs in the directory
    for run_dir in get_run_dirs(dir_name):
        print(run_dir)
        evaluate_run(run_dir)
    timer_end = time.time()
    print(f"Elapsed time: {timer_end - timer_start} seconds")
