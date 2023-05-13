import os
from platefno.util.conf import get_config
import matplotlib.pyplot as plt
import numpy as np


def plot_mse_per_timestep(
    mse_per_timestep,
    dir_name,
    fname="mse_per_step.pdf",
    average=False,
    highlight=None,
    linestyle="solid",
    label=None,
    **kwargs,
):
    """Plot the MSE per timestep for the three models. Assume we use the validation set from the run itself"""
    # Unpack the MSE per timestep
    val_gru_mse_per_step, val_rnn_mse_per_step, val_ref_mse_per_step = mse_per_timestep

    # Get fig from kwargs, or if not present assign to None
    fig = kwargs.get("fig", None)

    if fig is None:
        fig_width = 20
        figsize = (fig_width, fig_width * 0.75)
        fig = plt.figure(figsize=figsize)
        plt.rcParams.update(
            {
                "axes.titlesize": "large",
                "text.usetex": False,
                # "font.family": "serif",
                "font.size": 15,
                # "font.serif": ["Times"],
            }
        )

        gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.0)
        axs = gs.subplots(sharex="row", sharey="row")
    else:
        fig = kwargs["fig"]
        axs = kwargs["axs"]

    # if there is nsteps_plot in kwargs, use it
    if "nsteps_plot" in kwargs.keys():
        nsteps_plot = kwargs["nsteps_plot"]
    else:
        nsteps_plot = val_gru_mse_per_step.shape[1]

    if average:
        # Get the minimum at each time step
        val_gru_mse_per_step_min = val_gru_mse_per_step.min(axis=0, keepdims=True)
        val_rnn_mse_per_step_min = val_rnn_mse_per_step.min(axis=0, keepdims=True)
        val_ref_mse_per_step_min = val_ref_mse_per_step.min(axis=0, keepdims=True)
        # Get the maximum at each time step
        val_gru_mse_per_step_max = val_gru_mse_per_step.max(axis=0, keepdims=True)
        val_rnn_mse_per_step_max = val_rnn_mse_per_step.max(axis=0, keepdims=True)
        val_ref_mse_per_step_max = val_ref_mse_per_step.max(axis=0, keepdims=True)
        # Get the mean at each time step
        val_gru_mse_per_step = val_gru_mse_per_step.mean(axis=0, keepdims=True)
        val_rnn_mse_per_step = val_rnn_mse_per_step.mean(axis=0, keepdims=True)
        val_ref_mse_per_step = val_ref_mse_per_step.mean(axis=0, keepdims=True)

    for i in range(val_gru_mse_per_step.shape[0]):
        if i == highlight and not average:
            color = "red"
        else:
            color = "black"
        # Put the label only for the last line
        if i == val_gru_mse_per_step.shape[0] - 1:
            plot_label = label
        else:
            plot_label = None
        axs[0, 0].plot(
            val_gru_mse_per_step[i, :nsteps_plot, 0],
            color=color,
            linestyle=linestyle,
        )
        axs[0, 1].plot(
            val_rnn_mse_per_step[i, :nsteps_plot, 0],
            color=color,
            linestyle=linestyle,
        )
        axs[0, 2].plot(
            val_ref_mse_per_step[i, :nsteps_plot, 0],
            color=color,
            linestyle=linestyle,
            label=plot_label,
        )
        axs[1, 0].plot(
            val_gru_mse_per_step[i, :nsteps_plot, 1],
            color=color,
            linestyle=linestyle,
        )
        axs[1, 1].plot(
            val_rnn_mse_per_step[i, :nsteps_plot, 1],
            color=color,
            linestyle=linestyle,
        )
        axs[1, 2].plot(
            val_ref_mse_per_step[i, :nsteps_plot, 1],
            color=color,
            linestyle=linestyle,
            label=plot_label,
        )

        # If averagin plot the min and max as shaded area
        if average:
            axs[0, 0].fill_between(
                range(val_gru_mse_per_step.shape[1]),
                val_gru_mse_per_step_min[i, :, 0],
                val_gru_mse_per_step_max[i, :, 0],
                color=color,
                alpha=0.1,
            )
            axs[0, 1].fill_between(
                range(val_rnn_mse_per_step.shape[1]),
                val_rnn_mse_per_step_min[i, :, 0],
                val_rnn_mse_per_step_max[i, :, 0],
                color=color,
                alpha=0.1,
            )
            axs[0, 2].fill_between(
                range(val_ref_mse_per_step.shape[1]),
                val_ref_mse_per_step_min[i, :, 0],
                val_ref_mse_per_step_max[i, :, 0],
                color=color,
                alpha=0.1,
            )
            axs[1, 0].fill_between(
                range(val_gru_mse_per_step.shape[1]),
                val_gru_mse_per_step_min[i, :, 1],
                val_gru_mse_per_step_max[i, :, 1],
                color=color,
                alpha=0.1,
            )
            axs[1, 1].fill_between(
                range(val_rnn_mse_per_step.shape[1]),
                val_rnn_mse_per_step_min[i, :, 1],
                val_rnn_mse_per_step_max[i, :, 1],
                color=color,
                alpha=0.1,
            )
            axs[1, 2].fill_between(
                range(val_ref_mse_per_step.shape[1]),
                val_ref_mse_per_step_min[i, :, 1],
                val_ref_mse_per_step_max[i, :, 1],
                color=color,
                alpha=0.1,
            )

    # Set y limits
    max_mse_u = 0.2
    max_mse_v = 0.2

    limy_u = np.min(
        [
            val_gru_mse_per_step[..., 0].max(),
            val_ref_mse_per_step[..., 0].max(),
            val_rnn_mse_per_step[..., 0].max(),
            max_mse_u,
        ]
    )
    limy_v = np.min(
        [
            val_gru_mse_per_step[..., 1].max(),
            val_ref_mse_per_step[..., 1].max(),
            val_rnn_mse_per_step[..., 1].max(),
            max_mse_v,
        ]
    )

    # If axs was passed in kwargs and wasn't None, retrieve existing ylim top and compare, keeping the highest
    if "axs" in kwargs.keys() and kwargs["axs"] is not None:
        _, limy_u_old = axs[0, 0].get_ylim()
        _, limy_v_old = axs[1, 0].get_ylim()
        limy_u = np.max([limy_u, limy_u_old])
        limy_v = np.max([limy_v, limy_v_old])

    axs[0, 0].set_ylim([0, limy_u])
    axs[0, 1].set_ylim([0, limy_u])
    axs[0, 2].set_ylim([0, limy_u])
    axs[1, 0].set_ylim([0, limy_v])
    axs[1, 1].set_ylim([0, limy_v])
    axs[1, 2].set_ylim([0, limy_v])

    axs[0, 0].set(ylabel="displacement MSE")
    axs[1, 0].set(ylabel="velocity MSE")
    axs[1, 0].set(xlabel="Time step")
    axs[1, 1].set(xlabel="Time step")
    axs[1, 2].set(xlabel="Time step")

    axs[0, 0].set(title="FGRU")
    axs[0, 1].set(title="FRNN")
    axs[0, 2].set(title="Ref")

    axs[0, 2].legend()
    axs[1, 2].legend()

    fig.savefig(
        os.path.join(dir_name, "validation", fname),
        bbox_inches="tight",
    )

    return fig, axs

