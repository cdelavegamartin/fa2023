import numpy as np
import torch
import sys
import time
from platefno.solver.linear_plate_solver import LinearPlateSolver
from platefno.nn.fno_rnn import FNO_RNN_2d
from platefno.nn.fno_gru import FNO_GRU_2d
from platefno.nn.fno_ref import FNO_Markov_2d
import matplotlib.pyplot as plt


num_steps_train = 4
num_variations = 10
validation_split = 0.1

fs = 48000
dur = num_steps_train / fs
gamma = 1.0
kappa = 1.0
print(f"Simulation duration: {dur}")
t60 = ({"f": 100, "T60": 5}, {"f": 2000, "T60": 3})
aspect_ratio = 0.95
Nx = 40


if len(sys.argv) == 1:
    epochs = 1
else:
    epochs = int(sys.argv[1])
print("\r", f"Starting training for {epochs} epochs", end="")

width = 8

# Set torch device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
batch_size = 10

num_example_timesteps = 100

torch.manual_seed(0)  # Set seed for reproducibility
np.random.seed(0)
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
training_input = torch.zeros((num_variations, 1, solver.Nx, solver.Ny, 2))
training_output = torch.zeros(
    (num_variations, solver.numT - 1, solver.Nx, solver.Ny, 2)
)
# This is not the actual training shape, but the shape of the input and output of the solver
# The actual training shape is ((1-validation_split)*num_variations, num_timesteps, num_x, num_y, num_channels)
print("\n")
print(f"training input shape:{training_input.shape}")
print(f"training output shape:{training_output.shape}")
print("\n")


# Generate training data
for i in range(num_variations):
    ctr = 0.3 * np.random.rand(2) + 0.5
    wid = np.random.rand(1) * 0.15
    u0_max = np.random.rand(1)
    v0_max = 0.0
    w0 = solver.create_pluck(ctr, wid, u0_max, v0_max)
    u, v, _ = solver.solve(w0)
    training_input[i, :, :, :, :] = torch.tensor(
        np.stack([u[:, :, 0], v[:, :, 0]], axis=-1)
    ).unsqueeze(0)
    training_output[i, :, :, :, :] = torch.tensor(
        np.stack(
            [u[:, :, 1:].transpose(2, 0, 1), v[:, :, 1:].transpose(2, 0, 1)], axis=-1
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
        in_channels=2,
        out_channels=2,
        spatial_size_x=training_output.shape[2],
        spatial_size_y=training_output.shape[3],
        width=width,
    )
).to(device)
print(f"Instantiate RNN model")
model_rnn = torch.nn.DataParallel(
    FNO_RNN_2d(
        in_channels=2,
        out_channels=2,
        spatial_size_x=training_output.shape[2],
        spatial_size_y=training_output.shape[3],
        depth=3,
        width=width,
    )
).to(device)
print(f"Instantiate Ref model")
model_ref = torch.nn.DataParallel(
    FNO_Markov_2d(
        in_channels=2,
        out_channels=2,
        spatial_size_x=training_output.shape[2],
        spatial_size_y=training_output.shape[3],
        depth=3,
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
optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(dataloader)
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
directory = os.path.join("output", "2d_plate_" + now.strftime("%Y_%m_%d-%H_%M_%S"))
print("\n")
print(directory)
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
    torch.nn.functional.mse_loss(val_gru_out, validation_output).detach().cpu().numpy()
)
del val_gru_out
val_rnn_out = model_rnn(validation_input[:, 0, ...], validation_output.shape[1])
val_rnn_mse = (
    torch.nn.functional.mse_loss(val_rnn_out, validation_output).detach().cpu().numpy()
)
del val_rnn_out
val_ref_out = model_ref(validation_input[:, 0, ...], validation_output.shape[1])
val_ref_mse = (
    torch.nn.functional.mse_loss(val_ref_out, validation_output).detach().cpu().numpy()
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

# Pirnt the shape of output_sequence_gru
# print(f"output_sequence_gru.shape: {output_sequence_gru.shape}")

# Print the shape of the solver output, u
# print(f"u.shape: {u.shape}")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(
#     solver.X.transpose(),
#     solver.Y.transpose(),
#     u[..., -1],
#     cmap="viridis",
# )


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
