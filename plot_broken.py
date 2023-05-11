# import hydra
# from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

# import os
# import pathlib
# import pandas as pd
import matplotlib.pyplot as plt


from platefno.util.conf import get_config
from platefno.util.eval import (
    get_run_dirs,
    calculate_mse_per_timestep,
    load_models_from_dir,
    load_data,
    calculate_mse_crossval,
)
from platefno.util.plot import plot_mse_per_timestep


def evaluate_run_on_different_dataset(model_dir, data_dir):
    model_gru, model_rnn, model_ref = load_models_from_dir(model_dir)
    cfg = get_config(model_dir)

    # Set torch device if posible if not cpu
    if torch.cuda.is_available() and cfg.train.device == "cuda":
        device = torch.device(cfg.train.device)
    else:
        device = torch.device("cpu")

    validation_input, validation_output = load_data(data_dir)
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
    # Find the run that has the highest mse, and its index. Only for the ref model
    max_mse_u = np.max(val_ref_mse_per_step[..., 0])
    ind_u = np.unravel_index(
        np.argmax(val_ref_mse_per_step[..., 0], axis=None),
        val_ref_mse_per_step[..., 0].shape,
    )

    max_mse_v = np.max(val_ref_mse_per_step[..., 1])
    ind_v = np.unravel_index(
        np.argmax(val_ref_mse_per_step[..., 1], axis=None),
        val_ref_mse_per_step[..., 1].shape,
    )
    # Print the mse and the index
    print(f"Max MSE: {max_mse_u} at index {ind_u}")
    print(f"Max MSE: {max_mse_v} at index {ind_v}")

    # Plot the broken run initial condition with imshow. Only for the ref model
    fig_width = 237 / 72.27  # Latex columnwidth expressed in inches
    figsize = (fig_width, fig_width * 0.75)
    fig = plt.figure(figsize=figsize)
    plt.rcParams.update(
        {
            "axes.titlesize": "small",
            "text.usetex": False,
            # "font.family": "serif",
            "font.size": 9,
            # "font.serif": ["Times"],
        }
    )
    gs = fig.add_gridspec(2, 2, hspace=0.0, wspace=0.05)
    axs = gs.subplots()
    axs[0, 0].imshow(
        validation_input[ind_u[0], 0, :, :, 0].detach().cpu().numpy().transpose(),
        cmap="viridis",
        interpolation="none",
    )
    axs[0, 1].imshow(
        validation_input[ind_u[0] + 1, 0, :, :, 0].detach().cpu().numpy().transpose(),
        cmap="viridis",
        interpolation="none",
    )
    axs[1, 0].imshow(
        validation_input[ind_u[0] + 5, 0, :, :, 0].detach().cpu().numpy().transpose(),
        cmap="viridis",
        interpolation="none",
    )
    axs[1, 1].imshow(
        validation_input[ind_u[0] - 20, 0, :, :, 0].detach().cpu().numpy().transpose(),
        cmap="viridis",
        interpolation="none",
    )
    plt.show()

    # Remove the broken run from the dataset
    validation_input = torch.cat(
        [
            validation_input[: ind_u[0], ...],
            validation_input[ind_u[0] + 1 :, ...],
        ]
    )
    validation_output = torch.cat(
        [
            validation_output[: ind_u[0], ...],
            validation_output[ind_u[0] + 1 :, ...],
        ]
    )

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
    # Plot the MSE per step
    plot_mse_per_timestep(
        (val_gru_mse_per_step, val_rnn_mse_per_step, val_ref_mse_per_step),
        model_dir,
        "mse_per_step_fixed.pdf",
    )

    return


if __name__ == "__main__":
    evaluate_run_on_different_dataset(
        "/home/carlos/projects/platefno/output/pdeparamsweep-full/gamma_1.0-kappa_1.0/ic_pluck/seed_2",
        "/home/carlos/projects/platefno/output/pdeparamsweep-full/gamma_1.0-kappa_1.0/ic_pluck/seed_1",
    )
