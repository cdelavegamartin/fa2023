import os
from platefno.util.conf import get_config
import matplotlib.pyplot as plt


def plot_mse_per_timestep(mse_per_timestep, dir_name):
    """Plot the MSE per timestep for the three models. Assume we use the validation set from the run itself"""
    # Unpack the MSE per timestep
    val_gru_mse_per_step, val_rnn_mse_per_step, val_ref_mse_per_step = mse_per_timestep

    output_dir = dir_name
    cfg = get_config(dir_name)
    # Plot the MSE per step for the three models
    fig_width = 20
    figsize = (fig_width, fig_width * 0.75)
    fig = plt.figure(figsize=figsize)
    plt.rcParams.update(
        {
            "axes.titlesize": "large",
            "text.usetex": False,
            # "font.family": "serif",
            "font.size": 12,
            # "font.serif": ["Times"],
        }
    )

    gs = fig.add_gridspec(2, 3, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex="row", sharey="row")
    # axs = gs.subplots()
    for i in range(val_gru_mse_per_step.shape[0]):
        # choose color depending on the type of input
        if cfg.train.ic == "pluck":
            color = "blue"
        elif cfg.train.ic == "random":
            color = "red"
        elif cfg.train.ic == "mix":
            offset = cfg.train.num_variations - val_gru_mse_per_step.shape[0]

            if (i + offset) % 2 == 0:
                color = "blue"
            else:
                color = "red"

        axs[0, 0].plot(val_gru_mse_per_step[i, :, 0], color=color)
        axs[0, 1].plot(val_rnn_mse_per_step[i, :, 0], color=color)
        axs[0, 2].plot(val_ref_mse_per_step[i, :, 0], color=color)
        axs[1, 0].plot(val_gru_mse_per_step[i, :, 1], color=color)
        axs[1, 1].plot(val_rnn_mse_per_step[i, :, 1], color=color)
        axs[1, 2].plot(val_ref_mse_per_step[i, :, 1], color=color)
    axs[0, 0].set(title="FGRU")
    axs[0, 1].set(title="FRNN")
    axs[0, 2].set(title="Ref")

    axs[0, 0].set(ylabel="displacement MSE")
    axs[1, 0].set(ylabel="velocity MSE")
    axs[1, 0].set(xlabel="Time step")
    axs[1, 1].set(xlabel="Time step")
    axs[1, 2].set(xlabel="Time step")

    fig.savefig(
        os.path.join(output_dir, "validation", "mse_per_step.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    return
