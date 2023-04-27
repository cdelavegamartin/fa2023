import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import os
import time
import shutil

from platefno.solver.linear_plate_solver import LinearPlateSolver
from platefno.nn.fno_rnn import FNO_RNN_2d
from platefno.nn.fno_gru import FNO_GRU_2d
from platefno.nn.fno_ref import FNO_Markov_2d
import matplotlib.pyplot as plt

from platefno.util.conf import get_config, get_t60


def test(dirname):
    #######################################################################################################################
    # test the model
    output_dir = dirname
    cfg = get_config(dirname)
    device = torch.device(cfg.train.device)  # Set torch device
    torch.manual_seed(cfg.train.random_seed)  # Set seed for reproducibility
    np.random.seed(cfg.train.random_seed)

    # Solver and domain parameters
    fs = cfg.domain.sampling_rate
    t60 = get_t60(cfg)
    num_example_timesteps = 100
    display_timestep = num_example_timesteps - 1

    dur = (num_example_timesteps + 2) / fs
    # print(f"Simulation duration: {dur}")
    solver = LinearPlateSolver(
        SR=fs,
        TF=dur,
        gamma=cfg.solver.gamma,
        kappa=cfg.solver.kappa,
        t60=t60,
        aspect_ratio=cfg.domain.aspect_ratio,
        Nx=cfg.domain.nx,
    )

    # define parameters

    ctr = (0.4, 0.5)
    wid = 0.2
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
    # Instantiate the models
    # print(f"Instantiate GRU model")
    model_gru = torch.nn.DataParallel(
        FNO_GRU_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=solver.Nx,
            spatial_size_y=solver.Ny,
            width=cfg.nnarch.width,
        )
    ).to(device)
    # print(f"Instantiate RNN model")
    model_rnn = torch.nn.DataParallel(
        FNO_RNN_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=solver.Nx,
            spatial_size_y=solver.Ny,
            depth=cfg.nnarch.depth,
            width=cfg.nnarch.width,
        )
    ).to(device)
    # print(f"Instantiate Ref model")
    model_ref = torch.nn.DataParallel(
        FNO_Markov_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=solver.Nx,
            spatial_size_y=solver.Ny,
            depth=cfg.nnarch.depth,
            width=cfg.nnarch.width,
        )
    ).to(device)

    # Load the models
    # print(f"Load GRU model")
    model_gru.load_state_dict(
        torch.load(os.path.join(output_dir, "model", "model_gru.pth"))
    )
    # print(f"Load RNN model")
    model_rnn.load_state_dict(
        torch.load(os.path.join(output_dir, "model", "model_rnn.pth"))
    )
    # print(f"Load Ref model")
    model_ref.load_state_dict(
        torch.load(os.path.join(output_dir, "model", "model_ref.pth"))
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

    plot_norm = 1 / np.max(np.abs(u[:, :, 1:]))
    output_sequence_gru *= plot_norm
    output_sequence_rnn *= plot_norm
    output_sequence_ref *= plot_norm

    with open(output_dir + "/output_max.txt", "w") as f:
        f.write(
            " || ".join(
                [
                    f"GRU output max:{np.max(np.abs(output_sequence_gru[... , 0].detach().cpu().numpy())):.4f}",
                    f"RNN output max:{np.max(np.abs(output_sequence_rnn[... , 0].detach().cpu().numpy())):.4f}",
                    f"Ref output max:{np.max(np.abs(output_sequence_ref[... , 0].detach().cpu().numpy())):.4f}",
                    f"u output max:{np.max(np.abs(u)):.4f}",
                ]
            )
        )
        f.close()
    # get maximum displacement
    u_max = np.max(
        np.array(
            [
                np.max(np.abs(u)),
                np.max(np.abs(output_sequence_gru.detach().cpu().numpy())),
                np.max(np.abs(output_sequence_rnn.detach().cpu().numpy())),
                np.max(np.abs(output_sequence_ref.detach().cpu().numpy())),
            ]
        )
    )
    print("u_max = ", u_max)
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
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[0, 1].imshow(
        output_sequence_rnn[0, 0, :, :, 0].detach().cpu().numpy().transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[0, 2].imshow(
        output_sequence_ref[0, 0, :, :, 0].detach().cpu().numpy().transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[0, 3].imshow(
        u[..., 1].transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[1, 0].imshow(
        output_sequence_gru[0, display_timestep // 2, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[1, 1].imshow(
        output_sequence_rnn[0, display_timestep // 2, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[1, 2].imshow(
        output_sequence_ref[0, display_timestep // 2, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[1, 3].imshow(
        u[..., display_timestep // 2 + 1].transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[2, 0].imshow(
        output_sequence_gru[0, display_timestep, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[2, 1].imshow(
        output_sequence_rnn[0, display_timestep, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[2, 2].imshow(
        output_sequence_ref[0, display_timestep, :, :, 0]
        .detach()
        .cpu()
        .numpy()
        .transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
        interpolation="none",
    )
    axs[2, 3].imshow(
        u[..., display_timestep + 1].transpose(),
        cmap="viridis",
        vmin=-u_max,
        vmax=u_max,
        aspect=solver.hy / solver.hx,
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

    plt.savefig(output_dir + "/2d_plate_outputs.pdf", bbox_inches="tight")

    return


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    """Train 3 variants of the Fourier Neural Operator (FNO) to solve the 2D plate equation"""

    # Training parameters
    num_steps_train = cfg.train.num_steps_train
    num_variations = cfg.train.num_variations
    validation_split = cfg.train.val_split
    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size
    device = torch.device(cfg.train.device)  # Set torch device
    torch.manual_seed(cfg.train.random_seed)  # Set seed for reproducibility
    np.random.seed(cfg.train.random_seed)

    # Solver and domain parameters
    fs = cfg.domain.sampling_rate
    dur = num_steps_train / fs
    print(f"Simulation duration: {dur}")
    t60 = get_t60(cfg)

    #######################################################################################################################
    # The solver is used to generate the training data
    solver = LinearPlateSolver(
        SR=fs,
        TF=dur,
        gamma=cfg.solver.gamma,
        kappa=cfg.solver.kappa,
        t60=t60,
        aspect_ratio=cfg.domain.aspect_ratio,
        Nx=cfg.domain.nx,
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

    # Generate training data
    # The solver is used to generate the training data
    # The initial condition is either a pluck or a random displacement,
    # and the training dataset can be either or a mixture of both

    u0_max = 1.0  # Maximum displacement of the initial condition
    v0_max = 0.0  # Maximum velocity of the initial condition
    for i in range(num_variations):
        if cfg.train.ic == "pluck":
            ctr = (
                0.6 * np.random.rand(2) + 0.2
            )  # Center of the pluck, between 0.2 and 0.8, relative to the plate side lengths
            wid = (
                0.1 + np.random.rand(1) * 0.1
            )  # Width of the pluck, between 0.1 and 0.2

            w0 = solver.create_pluck(ctr, wid, u0_max, v0_max)
        elif cfg.train.ic == "random":
            w0 = solver.create_random_initial(u0_max=u0_max, v0_max=v0_max)
        elif cfg.train.ic == "mix":
            if i % 2 == 0:
                ctr = (
                    0.6 * np.random.rand(2) + 0.2
                )  # Center of the pluck, between 0.2 and 0.8, relative to the plate side lengths
                wid = (
                    0.1 + np.random.rand(1) * 0.1
                )  # Width of the pluck, between 0.1 and 0.2

                w0 = solver.create_pluck(ctr, wid, u0_max, v0_max)
            else:
                w0 = solver.create_random_initial(u0_max=u0_max, v0_max=v0_max)
        else:
            raise ValueError("Invalid type of initial condition")

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
    # print(f"normalization multiplier dimensions: {normalization_multiplier.shape}")
    training_input *= normalization_multiplier
    training_output *= normalization_multiplier

    # split the generated data into training and validation
    num_validation = int(np.ceil(validation_split * num_variations))
    validation_input = training_input[-num_validation:, ...]
    validation_output = training_output[-num_validation:, ...]
    training_input = training_input[:-num_validation, ...]
    training_output = training_output[:-num_validation, ...]

    # Instantiate the models
    # print(f"Instantiate GRU model")
    model_gru = torch.nn.DataParallel(
        FNO_GRU_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=solver.Nx,
            spatial_size_y=solver.Ny,
            width=cfg.nnarch.width,
        )
    ).to(device)
    # print(f"Instantiate RNN model")
    model_rnn = torch.nn.DataParallel(
        FNO_RNN_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=solver.Nx,
            spatial_size_y=solver.Ny,
            depth=cfg.nnarch.depth,
            width=cfg.nnarch.width,
        )
    ).to(device)
    # print(f"Instantiate Ref model")
    model_ref = torch.nn.DataParallel(
        FNO_Markov_2d(
            in_channels=cfg.solver.num_state_variables,
            out_channels=cfg.solver.num_state_variables,
            spatial_size_x=solver.Nx,
            spatial_size_y=solver.Ny,
            depth=cfg.nnarch.depth,
            width=cfg.nnarch.width,
        )
    ).to(device)

    # List of all parameters to be optimzed, also for gradient clipping
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
        params, lr=cfg.train.optimizer.lr, weight_decay=cfg.train.optimizer.weight_decay
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

    output_dir = (hydra.core.hydra_config.HydraConfig.get())["runtime"]["output_dir"]
    print("\n")
    print(output_dir)
    model_dir = os.path.join(output_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    plt.plot(loss_history)
    plt.savefig(output_dir + "/loss_history.pdf")

    # path = output_dir + "/model_gru.pt"
    torch.save(model_gru.state_dict(), os.path.join(model_dir, "model_gru.pth"))
    # path = output_dir + "/model_rnn.pt"
    torch.save(model_rnn.state_dict(), os.path.join(model_dir, "model_rnn.pth"))
    # path = output_dir + "/model_ref.pt"
    torch.save(model_ref.state_dict(), os.path.join(model_dir, "model_ref.pth"))
    # path = output_dir + "/norms.pt"
    torch.save(normalization_multiplier, os.path.join(model_dir, "norms.pt"))

    del input
    del output
    del dataloader
    del optimizer
    del params
    torch.cuda.empty_cache()
    #######################################################################################################################
    val_dir = os.path.join(output_dir, "validation")
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    validation_input = validation_input.to(device)
    validation_output = validation_output.to(device)

    val_gru_out = model_gru(validation_input[:, 0, ...], validation_output.shape[1])
    val_gru_mse = (
        torch.nn.functional.mse_loss(val_gru_out, validation_output)
        .detach()
        .cpu()
        .numpy()
    )
    np.save(val_dir + "/val_gru_out.npy", val_gru_out.detach().cpu().numpy())
    del val_gru_out

    val_rnn_out = model_rnn(validation_input[:, 0, ...], validation_output.shape[1])
    val_rnn_mse = (
        torch.nn.functional.mse_loss(val_rnn_out, validation_output)
        .detach()
        .cpu()
        .numpy()
    )
    np.save(val_dir + "/val_rnn_out.npy", val_rnn_out.detach().cpu().numpy())
    del val_rnn_out

    val_ref_out = model_ref(validation_input[:, 0, ...], validation_output.shape[1])
    val_ref_mse = (
        torch.nn.functional.mse_loss(val_ref_out, validation_output)
        .detach()
        .cpu()
        .numpy()
    )
    np.save(val_dir + "/val_ref_out.npy", val_ref_out.detach().cpu().numpy())
    del val_ref_out

    # Save validation results
    np.save(val_dir + "/validation_input.npy", validation_input.detach().cpu().numpy())
    np.save(
        val_dir + "/validation_output.npy", validation_output.detach().cpu().numpy()
    )

    np.save(val_dir + "/val_gru_mse.npy", val_gru_mse)
    np.save(val_dir + "/val_rnn_mse.npy", val_rnn_mse)
    np.save(val_dir + "/val_ref_mse.npy", val_ref_mse)

    with open(val_dir + "/validation.txt", "w") as f:
        f.write(
            f"GRU validation MSE:{val_gru_mse:.8f} || RNN validation MSE:{val_rnn_mse:.8f} || Ref validation MSE:{val_ref_mse:.8f}"
        )
        f.close()

    dirname = os.path.join(output_dir, "code")
    fname = __file__

    if not os.access(dirname, os.F_OK):
        os.mkdir(dirname, 0o700)

    shutil.copy(fname, dirname)

    # do test (this needs a flag probably)
    test(output_dir)

    return


if __name__ == "__main__":
    # print_hydra_config()
    import time

    timer_start = time.time()
    train()
    timer_end = time.time()
    print(f"Elapsed time: {timer_end - timer_start} seconds")
