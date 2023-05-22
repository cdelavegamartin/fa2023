import os
from platefno.util.conf import get_config
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_mse_per_timestep(
    mse_per_timestep,
    dir_name,
    fname="mse_per_step.pdf",
    average=False,
    highlight=None,
    linestyle="solid",
    label=None,
    plot_velocity=False,
    **kwargs,
):
    """Plot the MSE per timestep for the three models. Assume we use the validation set from the run itself"""
    debug = False

    # Unpack the MSE per timestep
    val_gru_mse_per_step, val_rnn_mse_per_step, val_ref_mse_per_step = mse_per_timestep

    # Get fig from kwargs, or if not present assign to None
    fig = kwargs.get("fig", None)

    if plot_velocity:
        n_rows = 2
    else:
        n_rows = 1

    if fig is None:
        fig_width = 20
        figsize = (fig_width, fig_width * 0.35 * n_rows)
        fig = plt.figure(figsize=figsize)
        plt.rcParams.update(
            {
                "axes.titlesize": 34,
                "axes.labelsize": 30,
                "font.size": 25,
            }
        )

        gs = fig.add_gridspec(n_rows, 3, hspace=0.05, wspace=0.0)
        axs = gs.subplots(sharex="row", sharey="row", squeeze=False)
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
        if plot_velocity:
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
            if plot_velocity:
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
    max_mse_u = kwargs.get("max_mse_u", 5.0)
    max_mse_v = kwargs.get("max_mse_v", 5.0)

    if debug:
        # Print the maximum MSE for each model
        print(f"GRU max MSE (u): {np.nanmax(val_gru_mse_per_step[..., 0], axis=1)}")
        print(f"RNN max MSE (u): {np.nanmax(val_rnn_mse_per_step[..., 0], axis=1)}")
        print(f"REF max MSE (u): {np.nanmax(val_ref_mse_per_step[..., 0], axis=1)}")

        print(f"GRU max MSE (v): {np.nanmax(val_gru_mse_per_step[..., 1], axis=1)}")
        print(f"RNN max MSE (v): {np.nanmax(val_rnn_mse_per_step[..., 1], axis=1)}")
        print(f"REF max MSE (v): {np.nanmax(val_ref_mse_per_step[..., 1], axis=1)}")

    limy_u = np.nanmin(
        [
            np.nanmax(
                [
                    np.nanmax(val_gru_mse_per_step[..., 0], axis=1),
                    np.nanmax(val_rnn_mse_per_step[..., 0], axis=1),
                    np.nanmax(val_ref_mse_per_step[..., 0], axis=1),
                ]
            ),
            max_mse_u,
        ]
    )
    if debug:
        # Print limy_u
        print(f"limy_u: {limy_u}")

    if plot_velocity:
        limy_v = np.nanmin(
            [
                np.nanmax(
                    [
                        np.nanmax(val_gru_mse_per_step[..., 1], axis=1),
                        np.nanmax(val_rnn_mse_per_step[..., 1], axis=1),
                        np.nanmax(val_ref_mse_per_step[..., 1], axis=1),
                    ]
                ),
                max_mse_v,
            ]
        )

    # If axs was passed in kwargs and wasn't None, retrieve existing ylim top and compare, keeping the highest
    if "axs" in kwargs.keys() and kwargs["axs"] is not None:
        _, limy_u_old = axs[0, 0].get_ylim()
        limy_u = np.max([limy_u, limy_u_old])

        if plot_velocity:
            _, limy_v_old = axs[1, 0].get_ylim()
            limy_v = np.max([limy_v, limy_v_old])
        if debug:
            # Print limy_u
            print(f"limy_u_old: {limy_u_old}")
            print(f"limy_u again: {limy_u}")

    axs[0, 0].set_ylim([0, limy_u])
    axs[0, 1].set_ylim([0, limy_u])
    axs[0, 2].set_ylim([0, limy_u])
    if plot_velocity:
        axs[1, 0].set_ylim([0, limy_v])
        axs[1, 1].set_ylim([0, limy_v])
        axs[1, 2].set_ylim([0, limy_v])

    # Plot vertical green line at max train steps if provided in kwargs
    train_step = kwargs.get("train_step", None)
    if train_step is not None:
        axs[0, 0].axvspan(0, kwargs["train_step"], color="blue", alpha=0.2)
        axs[0, 1].axvspan(0, kwargs["train_step"], color="blue", alpha=0.2)
        axs[0, 2].axvspan(0, kwargs["train_step"], color="blue", alpha=0.2)
        if plot_velocity:
            axs[1, 0].axvspan(0, kwargs["train_step"], color="blue", alpha=0.2)
            axs[1, 1].axvspan(0, kwargs["train_step"], color="blue", alpha=0.2)
            axs[1, 2].axvspan(0, kwargs["train_step"], color="blue", alpha=0.2)

    axs[0, 0].set(ylabel="displacement MSE")
    if plot_velocity:
        axs[1, 0].set(ylabel="velocity MSE")

    axs[n_rows - 1, 0].set(xlabel="Time step")
    axs[n_rows - 1, 1].set(xlabel="Time step")
    axs[n_rows - 1, 2].set(xlabel="Time step")

    axs[0, 0].set(title="FGRU")
    axs[0, 1].set(title="FRNN")
    axs[0, 2].set(title="Ref")

    # If there are xticks in kwargs use, those, else set them to 0, and half of the timesteps
    if "xticks" in kwargs.keys():
        xticks = kwargs["xticks"]
    else:
        xticks = [0, val_gru_mse_per_step.shape[1] // 2]

    for i in range(len(axs) - 1):
        for j in range(len(axs[0])):
            axs[i, j].set_xticks([])
    for i in range(len(axs[0])):
        axs[n_rows - 1, i].set_xticks(xticks)

    for i in range(len(axs)):
        axs[i, 2].legend()

    return fig, axs


# function to plot the ouput of the models and the ground truth for specific timesteps
def plot_run_output_single_ic(
    ground_truth_output, models_output, plot_steps=[], fs=None
):
    """This function expects the output of a single IC, so dims of ground_truth (timesteps, nx,ny, state_variables), Same for each of the model outputs"""
    output_gru, output_rnn, output_ref = models_output

    # if the ouptuts are torch.Tensor, convert to numpy
    if torch.is_tensor(ground_truth_output):
        ground_truth_output = ground_truth_output.detach().cpu().numpy()
    if torch.is_tensor(output_gru):
        output_gru = output_gru.detach().cpu().numpy()
    if torch.is_tensor(output_rnn):
        output_rnn = output_rnn.detach().cpu().numpy()
    if torch.is_tensor(output_ref):
        output_ref = output_ref.detach().cpu().numpy()

    # print the shapes of the outputs
    print(
        f"ground_truth_output.shape = {ground_truth_output.shape}, output_gru.shape = {output_gru.shape}, output_rnn.shape = {output_rnn.shape}, output_ref.shape = {output_ref.shape}"
    )

    if len(plot_steps) == 0:
        plot_steps = [0, output_gru.shape[0] // 2, output_gru.shape[0] - 1]
    # Check that all the plot_steps are valid
    # print the plot_steps
    print(f"plot_steps = {plot_steps}")
    assert np.all(
        np.array(plot_steps) < output_gru.shape[0]
    ), "plot_steps must be smaller than the number of timesteps in the output"

    # get the aspect ratio
    aspect_ratio = (ground_truth_output.shape[2] - 1) / (
        ground_truth_output.shape[1] - 1
    )

    # Calculate limits for the colorbar
    max_val = np.max(ground_truth_output[..., 0])
    min_val = np.min(ground_truth_output[..., 0])

    fig_width = 20
    figsize = (fig_width, fig_width * (len(plot_steps) / 4))
    fig = plt.figure(figsize=figsize)
    plt.rcParams.update(
        {
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "text.usetex": False,
            "font.size": 15,
        }
    )

    gs = fig.add_gridspec(len(plot_steps), 4, hspace=0.0, wspace=0.05)
    axs = gs.subplots(sharex="row", sharey=True)

    for i, step in enumerate(plot_steps):
        # Plot ground truth
        axs[i, 0].imshow(
            output_gru[step, :, :, 0].transpose(),
            cmap="viridis",
            aspect=aspect_ratio,
            vmin=-max_val,
            vmax=max_val,
            interpolation="none",
        )
        axs[i, 1].imshow(
            output_rnn[step, :, :, 0].transpose(),
            cmap="viridis",
            aspect=aspect_ratio,
            vmin=-max_val,
            vmax=max_val,
            interpolation="none",
        )
        axs[i, 2].imshow(
            output_ref[step, :, :, 0].transpose(),
            cmap="viridis",
            aspect=aspect_ratio,
            vmin=-max_val,
            vmax=max_val,
            interpolation="none",
        )
        axs[i, 3].imshow(
            ground_truth_output[step, :, :, 0].transpose(),
            cmap="viridis",
            aspect=aspect_ratio,
            vmin=-max_val,
            vmax=max_val,
            interpolation="none",
        )
        if fs is not None:
            axs[i, 0].set(ylabel=f"Step = {(step)}")

    axs[0, 0].set(title="FGRU")
    axs[0, 1].set(title="FRNN")
    axs[0, 2].set(title="REF")
    axs[0, 3].set(title="Truth")

    axs[len(plot_steps) - 1, 3].set(xlabel="x (/m)")
    axs[len(plot_steps) - 1, 3].yaxis.set_label_position("right")
    axs[len(plot_steps) - 1, 3].set(ylabel="y (/m)")

    for i in range(len(axs)):
        for j in range(len(axs[0])):
            axs[i, j].get_images()[0].set_clim(-max_val, max_val)
            axs[i, j].label_outer()
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    return fig, axs
