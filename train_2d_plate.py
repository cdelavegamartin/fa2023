import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import os
import time
import shutil

from hydra.utils import to_absolute_path

from platefno.data import channel_stats, split, standardize
from platefno.nn.models import build_models
import matplotlib.pyplot as plt



@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    """Train 3 variants of the Fourier Neural Operator (FNO) to solve the 2D plate equation"""

    # Training parameters
    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size
    device = torch.device(cfg.train.device)  # Set torch device
    torch.manual_seed(cfg.train.random_seed)  # Set seed for reproducibility
    np.random.seed(cfg.train.random_seed)

    # Dataset made by generate_dataset.py: (trajectories, time steps, x, y, channels)
    data = np.load(to_absolute_path(cfg.data.path), mmap_mode="r")
    training, validation = split(data, cfg.train.val_split)
    num_channels = data.shape[-1]
    mean = list(cfg.data.mean) if cfg.data.mean is not None else [0.0] * num_channels
    std = list(cfg.data.std) if cfg.data.std is not None else channel_stats(training)[1].tolist()
    print(f"Standardisation: mean {mean}, std {std}")
    training, validation = (standardize(d, mean, std) for d in (training, validation))
    # The first step is the input, the following steps the target
    training_input, training_output = training[:, :1], training[:, 1:]
    validation_input, validation_output = validation[:, :1], validation[:, 1:]
    normalization_multiplier = 1 / torch.tensor(std)  # as used by the evaluation scripts
    size_x, size_y = data.shape[2:4]

    # Instantiate the models
    model_gru, model_rnn, model_ref = build_models(cfg, size_x, size_y, device)

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
    OmegaConf.save(
        OmegaConf.create({"path": cfg.data.path, "mean": list(mean), "std": list(std)}),
        os.path.join(model_dir, "data_stats.yaml"),
    )

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

    val_mse = {}
    with torch.no_grad():
        for name, model in zip(("gru", "rnn", "ref"), (model_gru, model_rnn, model_ref)):
            val_out = model(validation_input[:, 0, ...], validation_output.shape[1])
            val_mse[name] = (
                torch.nn.functional.mse_loss(val_out, validation_output).cpu().numpy()
            )
            np.save(val_dir + f"/val_{name}_out.npy", val_out.cpu().numpy())
            del val_out
    val_gru_mse, val_rnn_mse, val_ref_mse = val_mse.values()

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

    return


if __name__ == "__main__":
    # print_hydra_config()
    import time

    timer_start = time.time()
    train()
    timer_end = time.time()
    print(f"Elapsed time: {timer_end - timer_start} seconds")
