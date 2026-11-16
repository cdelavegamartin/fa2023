"""Simulate the plate dataset of a configuration and save it as one .npy array.

Writes <data.path> (trajectories, time steps, x, y, channels), unnormalised, and
<data.path stem>.yaml with the settings that define it and the per-channel mean
and std of its training split.
"""

import os

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from platefno.data import channel_stats, simulate, split


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    path = to_absolute_path(cfg.data.path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = simulate(cfg)
    np.save(path, data)

    training, _ = split(data, cfg.train.val_split)
    mean, std = channel_stats(training)
    info = {
        "shape": list(data.shape),
        "training_split": {"mean": mean.tolist(), "std": std.tolist()},
        "domain": cfg.domain,
        "solver": cfg.solver,
        "train": {
            key: cfg.train[key]
            for key in ("num_variations", "num_steps_train", "val_split", "ic", "random_seed")
        },
    }
    OmegaConf.save(OmegaConf.create(info), os.path.splitext(path)[0] + ".yaml")
    print(f"Dataset saved: {path} {data.shape}")


if __name__ == "__main__":
    main()
