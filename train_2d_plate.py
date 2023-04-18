import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import sys
import time
from platefno.solver.linear_plate_solver import LinearPlateSolver
from platefno.nn.fno_rnn import FNO_RNN_2d
from platefno.nn.fno_gru import FNO_GRU_2d
from platefno.nn.fno_ref import FNO_Markov_2d
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    num_steps_train = cfg.train.num_steps_train
    num_variations = cfg.train.num_variations
    validation_split = cfg.train.val_split

    fs = cfg.domain.sampling_rate
    dur = num_steps_train / fs
    gamma = cfg.solver.gamma
    kappa = cfg.solver.kappa
    print(f"Simulation duration: {dur}")
    # Set T60 depending on the damping configuration
    if cfg.solver.damping.name == "nodamping":
        t60 = None
    elif cfg.solver.damping.name == "constant":
        t60 = cfg.solver.damping.T60
    elif cfg.solver.damping.name == "freqdependent":
        t60 = (
            {"f": cfg.solver.damping.f1, "T60": cfg.solver.damping.T60f1},
            {"f": cfg.solver.damping.f2, "T60": cfg.solver.damping.T60f2},
        )
    else:
        raise ValueError("Invalid damping configuration")
    aspect_ratio = cfg.domain.aspect_ratio
    Nx = cfg.domain.nx

    epochs = cfg.train.epochs
    print("\r", f"Starting training for {epochs} epochs", end="")

    width = cfg.nnarch.width

    # Set torch device
    device = torch.device(cfg.train.device)
    batch_size = cfg.train.batch_size

    num_example_timesteps = 100

    torch.manual_seed(cfg.train.random_seed)  # Set seed for reproducibility
    np.random.seed(cfg.train.random_seed)
    #######################################################################################################################
    # The solver is used to generate the training data
    solver = LinearPlateSolver(
        SR=fs,
        TF=dur,
        gamma=gamma,
        kappa=kappa,
        t60=t60,
        aspect_ratio=aspect_ratio,
        Nx=Nx,
    )
    # Last dimension is for the 2 channels (displacement, velocity)
    training_input = torch.zeros(
        (num_variations, 1, solver.Nx, solver.Ny, cfg.solver.num_state_variables)
    )
    training_output = torch.zeros(
        (
            num_variations,
            solver.numT - 1,
            solver.Nx,
            solver.Ny,
            cfg.solver.num_state_variables,
        )
    )
    # This is not the actual training shape, but the shape of the input and output of the solver
    # The actual training shape is ((1-validation_split)*num_variations, num_timesteps, num_x, num_y, num_channels)
    print("\n")
    print(f"training input shape:{training_input.shape}")
    print(f"training output shape:{training_output.shape}")
    print("\n")

    # Generate training data
    for i in range(num_variations):
        ctr = (
            0.6 * np.random.rand(2) + 0.2
        )  # Center of the pluck, between 0.2 and 0.8, relative to the plate side lengths
        wid = np.random.rand(1) * 0.15  # Width of the pluck, between 0 and 0.15
        u0_max = np.random.rand(1)  # Maximum displacement of the pluck
        v0_max = 0.0  # Maximum velocity of the pluck
        w0 = solver.create_pluck(ctr, wid, u0_max, v0_max)
        u, v, _ = solver.solve(w0)
        training_input[i, :, :, :, :] = torch.tensor(
            np.stack([u[:, :, 0], v[:, :, 0]], axis=-1)
        ).unsqueeze(0)
        training_output[i, :, :, :, :] = torch.tensor(
            np.stack(
                [u[:, :, 1:].transpose(2, 0, 1), v[:, :, 1:].transpose(2, 0, 1)],
                axis=-1,
            )
        ).unsqueeze(0)

    normalization_multiplier = 1 / torch.std(training_output, dim=(0, 1, 2, 3))
    print(f"normalization multiplier dimensions: {normalization_multiplier.shape}")
    training_input *= normalization_multiplier
    training_output *= normalization_multiplier

    # split the generated data into training and validation
    num_validation = int(np.ceil(validation_split * num_variations))
    validation_input = training_input[-num_validation:, ...]
    validation_output = training_output[-num_validation:, ...]
    training_input = training_input[:-num_validation, ...]
    training_output = training_output[:-num_validation, ...]

    learning_rate = 1e-4
    print(f"Instantiate GRU model")
    model_gru = torch.nn.DataParallel(
        FNO_GRU_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=training_output.shape[2],
            spatial_size_y=training_output.shape[3],
            width=width,
        )
    ).to(device)
    print(f"Instantiate RNN model")
    model_rnn = torch.nn.DataParallel(
        FNO_RNN_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=training_output.shape[2],
            spatial_size_y=training_output.shape[3],
            depth=cfg.nnarch.depth,
            width=width,
        )
    ).to(device)
    print(f"Instantiate Ref model")
    model_ref = torch.nn.DataParallel(
        FNO_Markov_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=training_output.shape[2],
            spatial_size_y=training_output.shape[3],
            depth=cfg.nnarch.depth,
            width=width,
        )
    ).to(device)

    params = (
        list(model_gru.parameters())
        + list(model_rnn.parameters())
        + list(model_ref.parameters())
    )
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(training_input, training_output),
        batch_size=batch_size,
        shuffle=True,
    )
    # Assert that the optimizer is AdamW, otherwise the weight decay is not applied to the bias terms
    assert cfg.train.optimizer.name == "AdamW"
    optimizer = torch.optim.AdamW(
        params, lr=learning_rate, weight_decay=cfg.train.optimizer.weight_decay
    )
    # Assert that the scheduler is OneCycleLR, otherwise the learning rate is not increased linearly
    assert cfg.train.scheduler.name == "OneCycleLR"
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.train.scheduler.max_lr,
        epochs=epochs,
        steps_per_epoch=len(dataloader),
    )

    loss_history = np.zeros((epochs, 3))

    print(f"Starting training loop...", "\n")
    for ep in range(epochs):
        tic = time.time()
        for input, output in dataloader:
            input, output = input.to(device), output.to(device)
            optimizer.zero_grad()
            model_input = input[:, 0, ...]

            pred_gru = model_gru(model_input, num_time_steps=training_output.shape[1])
            loss_gru = torch.log10(torch.nn.functional.mse_loss(pred_gru, output))
            loss_gru.backward()
            del pred_gru

            pred_rnn = model_rnn(model_input, num_time_steps=training_output.shape[1])
            loss_rnn = torch.log10(torch.nn.functional.mse_loss(pred_rnn, output))
            loss_rnn.backward()
            del pred_rnn

            pred_ref = model_ref(model_input, num_time_steps=training_output.shape[1])
            loss_ref = torch.log10(torch.nn.functional.mse_loss(pred_ref, output))
            loss_ref.backward()
            del pred_ref
            # Clipping is applied to all the parameters, individually.
            torch.nn.utils.clip_grad_norm_(params, 1)
            optimizer.step()
            scheduler.step()
        loss_history[ep, 0] = np.power(10, loss_gru.detach().cpu().numpy())
        loss_history[ep, 1] = np.power(10, loss_rnn.detach().cpu().numpy())
        loss_history[ep, 2] = np.power(10, loss_ref.detach().cpu().numpy())
        elapsed = time.time() - tic
        time_remaining = elapsed * (epochs - ep) / (60.0 * 60.0)
        print(
            "\r",
            f"epochs:{ep}, gru_loss:{loss_history[ep,0]:.5f}, rnn_loss:{loss_history[ep,1]:.5f}, ref_loss:{loss_history[ep,2]:.5f}, epoch_time(s):{elapsed:.2f}, time_remaining(hrs):{time_remaining:.2f}",
            end="",
        )

    from datetime import datetime
    import os

    now = datetime.now()
    directory = hydra_cfg["runtime"]["output_dir"]
    print("\n")
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.plot(loss_history)
    plt.savefig(directory + "/loss_history.pdf")

    path = directory + "/model_gru.pt"
    torch.save(model_gru, path)
    path = directory + "/model_rnn.pt"
    torch.save(model_rnn, path)
    path = directory + "/model_ref.pt"
    torch.save(model_ref, path)
    path = directory + "/norms.pt"
    torch.save(normalization_multiplier, path)

    del input
    del output
    del dataloader
    del optimizer
    del params
    torch.cuda.empty_cache()
    #######################################################################################################################
    validation_input = validation_input.to(device)
    validation_output = validation_output.to(device)

    val_gru_out = model_gru(validation_input[:, 0, ...], validation_output.shape[1])
    val_gru_mse = (
        torch.nn.functional.mse_loss(val_gru_out, validation_output)
        .detach()
        .cpu()
        .numpy()
    )
    del val_gru_out
    val_rnn_out = model_rnn(validation_input[:, 0, ...], validation_output.shape[1])
    val_rnn_mse = (
        torch.nn.functional.mse_loss(val_rnn_out, validation_output)
        .detach()
        .cpu()
        .numpy()
    )
    del val_rnn_out
    val_ref_out = model_ref(validation_input[:, 0, ...], validation_output.shape[1])
    val_ref_mse = (
        torch.nn.functional.mse_loss(val_ref_out, validation_output)
        .detach()
        .cpu()
        .numpy()
    )
    del val_ref_out

    with open(directory + "/validation.txt", "w") as f:
        f.write(
            f"GRU validation MSE:{val_gru_mse:.8f} || RNN validation MSE:{val_rnn_mse:.8f} || Ref validation MSE:{val_ref_mse:.8f}"
        )
        f.close()

    #######################################################################################################################
    display_timestep = num_example_timesteps - 1

    dur = (num_example_timesteps + 2) / fs
    solver = LinearPlateSolver(
        SR=fs,
        TF=dur,
        gamma=gamma,
        kappa=kappa,
        t60=t60,
        aspect_ratio=aspect_ratio,
        Nx=Nx,
    )

    # define parameters

    ctr = (0.8, 0.5)
    wid = 0.1
    u0_max = 1.0
    v0_max = 0

    w0 = solver.create_pluck(ctr, wid, u0_max, v0_max)
    u, v, _ = solver.solve(w0)
    model_input = (
        torch.tensor(
            np.stack(
                [
                    u[:, :, 0],
                    v[:, :, 0],
                ],
                axis=-1,
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    normalization_multiplier = 1 / torch.tensor(np.stack([u, v], axis=-1)).std(
        dim=(
            0,
            1,
            2,
        )
    )

    # Print the shape of model input and its normalization multiplier
    # print(f"model_input.shape before normalization: {model_input.shape}")
    # print(f"normalization_multiplier.shape: {normalization_multiplier.shape}")

    model_input *= normalization_multiplier.to(device)
    # print(f"model_input.shape after normalization: {model_input.shape}")

    # Print type of model_input
    # print(f"model_input.dtype test: {model_input.dtype}")

    output_sequence_gru = model_gru(model_input, num_example_timesteps)
    output_sequence_rnn = model_rnn(model_input, num_example_timesteps)
    output_sequence_ref = model_ref(model_input, num_example_timesteps)

    plot_norm = 1 / np.max(np.abs(u[:, :, 10:]))
    output_sequence_gru *= plot_norm
    output_sequence_rnn *= plot_norm
    output_sequence_ref *= plot_norm

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

    gs = fig.add_gridspec(3, 4, hspace=0.0, wspace=0.05)
    axs = gs.subplots(sharex="row", sharey=True)
    axs[0, 0].imshow(
        output_sequence_gru[0, 0, :, :, 0].detach().cpu().numpy().transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[0, 1].imshow(
        output_sequence_rnn[0, 0, :, :, 0].detach().cpu().numpy().transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[0, 2].imshow(
        output_sequence_ref[0, 0, :, :, 0].detach().cpu().numpy().transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[0, 3].imshow(
        u[..., 1].transpose(), cmap="viridis", aspect="equal", interpolation="none"
    )
    axs[1, 0].imshow(
        output_sequence_gru[0, display_timestep // 2, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[1, 1].imshow(
        output_sequence_rnn[0, display_timestep // 2, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[1, 2].imshow(
        output_sequence_ref[0, display_timestep // 2, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[1, 3].imshow(
        u[..., display_timestep // 2 + 1].transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[2, 0].imshow(
        output_sequence_gru[0, display_timestep, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[2, 1].imshow(
        output_sequence_rnn[0, display_timestep, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[2, 2].imshow(
        output_sequence_ref[0, display_timestep, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )
    axs[2, 3].imshow(
        u[..., display_timestep + 1].transpose(),
        cmap="viridis",
        aspect="equal",
        interpolation="none",
    )

    axs[0, 0].set(title="FGRU")
    axs[0, 1].set(title="FRNN")
    axs[0, 2].set(title="Ref.")
    axs[0, 3].set(title="Truth")

    axs[0, 0].set(ylabel="0 ms")
    axs[1, 0].set(ylabel=f"{((display_timestep // 2)/fs)*1000:.1f} ms")
    axs[2, 0].set(ylabel=f"{(display_timestep/fs)*1000:.1f} ms")

    axs[2, 3].set(xlabel="x (/m)")
    axs[2, 3].yaxis.set_label_position("right")
    axs[2, 3].set(ylabel="y (/m)")

    for i in range(len(axs)):
        for j in range(len(axs[0])):
            axs[i, j].get_images()[0].set_clim(-1, 1)
            axs[i, j].label_outer()
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    plt.savefig(directory + "/2d_plate_outputs.pdf", bbox_inches="tight")
    return


if __name__ == "__main__":
    # print_hydra_config()
    import time

    timer_start = time.time()
    train()
    timer_end = time.time()
    print(f"Elapsed time: {timer_end - timer_start} seconds")
