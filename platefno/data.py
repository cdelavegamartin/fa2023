"""Generation, splitting and standardisation of the plate datasets.

A dataset is one raw (unnormalised) array of shape
(trajectories, time steps, x, y, channels), saved as .npy by generate_dataset.py.
"""

import numpy as np
import torch

from platefno.solver.linear_plate_solver import LinearPlateSolver
from platefno.util.conf import get_t60


def make_solver(cfg):
    """LinearPlateSolver for the domain, solver and training length of the config."""
    fs = cfg.domain.sampling_rate
    return LinearPlateSolver(
        SR=fs,
        TF=cfg.train.num_steps_train / fs,
        gamma=cfg.solver.gamma,
        kappa=cfg.solver.kappa,
        t60=get_t60(cfg),
        aspect_ratio=cfg.domain.aspect_ratio,
        Nx=cfg.domain.nx,
    )


def random_pluck(solver, u0_max, v0_max):
    ctr = 0.6 * np.random.rand(2) + 0.2  # centre between 0.2 and 0.8 of the side lengths
    wid = 0.1 + np.random.rand(1) * 0.1  # width between 0.1 and 0.2
    return solver.create_pluck(ctr, wid, u0_max, v0_max)


def initial_condition(solver, ic, index, u0_max=1.0, v0_max=0.0):
    """Pluck, random field, or alternating mix of both."""
    if ic == "pluck" or (ic == "mix" and index % 2 == 0):
        return random_pluck(solver, u0_max, v0_max)
    if ic in ("random", "mix"):
        return solver.create_random_initial(u0_max=u0_max, v0_max=v0_max)
    raise ValueError("Invalid type of initial condition")


def simulate(cfg):
    """Simulate cfg.train.num_variations trajectories, seeding numpy with cfg.train.random_seed.

    Returns a float32 array (trajectories, time steps, x, y, [displacement, velocity]).
    """
    np.random.seed(cfg.train.random_seed)
    solver = make_solver(cfg)
    n = cfg.train.num_variations
    data = np.zeros((n, solver.numT, solver.Nx, solver.Ny, 2), dtype=np.float32)
    for i in range(n):
        u, v, _ = solver.solve(initial_condition(solver, cfg.train.ic, i))
        data[i] = np.stack([u, v], axis=-1).transpose(2, 0, 1, 3)
    return data


def split(data, val_split):
    """Training and validation trajectories; the last ceil(val_split * n) validate."""
    num_validation = int(np.ceil(val_split * data.shape[0]))
    return data[:-num_validation], data[-num_validation:]


def channel_stats(data):
    """Per-channel mean and standard deviation over trajectories, time and space."""
    axes = tuple(range(data.ndim - 1))
    return data.mean(axis=axes), data.std(axis=axes)


def standardize(data, mean, std):
    """(data - mean) / std per channel, as a float32 torch tensor."""
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return torch.from_numpy(((data - mean) / std).astype(np.float32))
