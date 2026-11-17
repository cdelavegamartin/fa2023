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
from platefno.nn.training import clip_groups, train_step
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

    output_dir = (hydra.core.hydra_config.HydraConfig.get())["runtime"]["output_dir"]

    # Instantiate the models; each model has its own seed, so its initial weights do
    # not depend on which other models a run trains
    all_models = build_models(cfg, size_x, size_y, device, seed=cfg.train.random_seed)
    models = {name: all_models[name] for name in cfg.train.models}

    # The batch order depends only on the dataset seed, not on the models or their seed
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(training_input, training_output),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.data.seed),
    )
    # One optimiser and scheduler per model, so that models can train in separate runs
    assert cfg.train.optimizer.name == "AdamW"
    assert cfg.train.scheduler.name == "OneCycleLR"
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.optimizer.lr,
            weight_decay=cfg.train.optimizer.weight_decay,
        )
        for name, model in models.items()
    }
    schedulers = [
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.train.scheduler.max_lr,
            epochs=epochs,
            steps_per_epoch=len(dataloader),
        )
        for optimizer in optimizers.values()
    ]
    # clip_jointly clips the concatenated gradients of all the models of the run, as
    # in the original training; otherwise each model is clipped on its own
    groups = clip_groups(models, cfg.train.clip_jointly)

    loss_history = {name: np.zeros(epochs) for name in models}

    print(f"Training {list(models)}, clip_jointly={cfg.train.clip_jointly}", "\n")
    for ep in range(epochs):
        tic = time.time()
        for input, output in dataloader:
            input, output = input.to(device), output.to(device)
            mse = train_step(models, optimizers, groups, input[:, 0, ...], output)
            for scheduler in schedulers:
                scheduler.step()
        for name in models:
            loss_history[name][ep] = mse[name]  # last batch of the epoch
        elapsed = time.time() - tic
        losses = ", ".join(f"{name}_loss:{h[ep]:.5f}" for name, h in loss_history.items())
        print(
            "\r",
            f"epochs:{ep}, {losses}, epoch_time(s):{elapsed:.2f}, "
            f"time_remaining(hrs):{elapsed * (epochs - ep) / 3600:.2f}",
            end="",
        )

    print("\n")
    print(output_dir)
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    # Per-model files, so that runs training different models can share a directory
    for name, model in models.items():
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_{name}.pth"))
        np.save(os.path.join(output_dir, f"loss_history_{name}.npy"), loss_history[name])
    plt.semilogy(np.stack(list(loss_history.values()), axis=1), label=list(models))
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"loss_history_{'_'.join(models)}.pdf"))
    torch.save(normalization_multiplier, os.path.join(model_dir, "norms.pt"))
    OmegaConf.save(
        OmegaConf.create({"path": cfg.data.path, "mean": list(mean), "std": list(std)}),
        os.path.join(model_dir, "data_stats.yaml"),
    )

    del input, output, dataloader, optimizers
    torch.cuda.empty_cache()
    #######################################################################################################################
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    # Validation data, normalised as the models see it, for the evaluation scripts
    np.save(os.path.join(val_dir, "validation_input.npy"), validation_input.numpy())
    np.save(os.path.join(val_dir, "validation_output.npy"), validation_output.numpy())

    validation_input = validation_input.to(device)
    validation_output = validation_output.to(device)
    with torch.no_grad():
        for name, model in models.items():
            val_out = model(validation_input[:, 0, ...], validation_output.shape[1])
            val_mse = torch.nn.functional.mse_loss(val_out, validation_output).cpu().numpy()
            np.save(os.path.join(val_dir, f"val_{name}_out.npy"), val_out.cpu().numpy())
            np.save(os.path.join(val_dir, f"val_{name}_mse.npy"), val_mse)
            with open(os.path.join(val_dir, f"validation_{name}.txt"), "w") as f:
                f.write(f"{name.upper()} validation MSE:{val_mse:.8f}")
            del val_out

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
