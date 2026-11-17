"""Generation, splitting and standardisation of the datasets.

A dataset is one raw (unnormalised) array of shape
(trajectories, time steps, x, y, channels), saved as .npy by generate_dataset.py.
solver.name selects the system: "linearplate" (finite-difference stiff plate,
channels displacement and velocity) or "wave2d" (FTM 2D wave equation of Parker et
al., channels pressure and the two particle velocities).
"""

import numpy as np
import torch

from platefno.solver.linear_plate_solver import LinearPlateSolver
from platefno.solver.wave_solver_2d import WaveSolver2D
from platefno.util.conf import get_t60


# Settings of conf/solver/wave2d.yaml, used by the plate's impulse initial conditions
WAVE_DEFAULTS = {"c0": 340, "modes": 25}
# Plate initial conditions taken from the wave2d solver
WAVE_ICS = ("impulse", "impulse_random")


def make_wave_solver(cfg, num_steps):
    """FTM 2D wave solver on the grid of the config, for num_steps time steps.

    Its grid, linspace(0, 1, nx) x linspace(0, aspect_ratio, ny), is the plate's grid.
    """
    fs = cfg.domain.sampling_rate
    return WaveSolver2D(
        Fs=fs,
        dur=num_steps / fs,
        lx=1.0,
        ly=cfg.domain.aspect_ratio,
        spatial_delta=1.0 / cfg.domain.nx,
        c0=cfg.solver.get("c0", WAVE_DEFAULTS["c0"]),
        modes=cfg.solver.get("modes", WAVE_DEFAULTS["modes"]),
    )


def make_solver(cfg, num_steps=None):
    """Solver of the config for num_steps time steps (default: the training length).

    The domain is [0, 1] x [0, domain.aspect_ratio] with domain.nx points along x.
    """
    fs = cfg.domain.sampling_rate
    num_steps = num_steps or cfg.train.num_steps_train
    if cfg.solver.name == "wave2d":
        return make_wave_solver(cfg, num_steps)
    duration = num_steps / fs
    return LinearPlateSolver(
        SR=fs,
        TF=duration,
        gamma=cfg.solver.gamma,
        kappa=cfg.solver.kappa,
        t60=get_t60(cfg),
        aspect_ratio=cfg.domain.aspect_ratio,
        Nx=cfg.domain.nx,
    )


def grid_size(cfg):
    """(x, y) grid points of the config's solver."""
    solver = make_solver(cfg, num_steps=2)
    if cfg.solver.name == "wave2d":
        return solver.numXs, solver.numYs
    return solver.Nx, solver.Ny


def random_pluck(solver, u0_max, v0_max):
    ctr = 0.6 * np.random.rand(2) + 0.2  # centre between 0.2 and 0.8 of the side lengths
    wid = 0.1 + np.random.rand(1) * 0.1  # width between 0.1 and 0.2
    return solver.create_pluck(ctr, wid, u0_max, v0_max)


def initial_condition(solver, ic, index, u0_max=1.0, v0_max=0.0):
    """Plate: pluck, random field, or alternating mix of both."""
    if ic == "pluck" or (ic == "mix" and index % 2 == 0):
        return random_pluck(solver, u0_max, v0_max)
    if ic in ("random", "mix"):
        return solver.create_random_initial(u0_max=u0_max, v0_max=v0_max)
    raise ValueError("Invalid type of initial condition")


def wave_initial_condition(solver, ic, index, num):
    """2D wave: impulse at a random position or random field; "impulse_random" gives
    impulses for the first half and random fields for the second (as in Parker et al.)."""
    if ic == "impulse" or (ic == "impulse_random" and index < num // 2):
        return solver.create_impulse(np.random.rand(1), np.random.rand(1))
    if ic in ("random", "impulse_random"):
        return solver.create_random_initial()
    raise ValueError("Invalid type of initial condition")


def plate_wave_initial_condition(wave, ic, index, num, u0_max=1.0):
    """Plate state (displacement, velocity) from the initial pressure of the wave2d
    solver, with the random draws of the wave2d dataset.

    The wave solver starts with zero particle velocity, so dp/dt = 0 at t = 0, and the
    plate starts at rest: u0 = p0 with the boundary set to 0 (u = 0 on the plate's
    edges), scaled to a maximum of u0_max.
    """
    _, _, pressure, _, _ = wave.solve(wave_initial_condition(wave, ic, index, num))
    u0 = pressure[:, :, 0].astype(np.float64)
    u0[[0, -1], :] = 0.0
    u0[:, [0, -1]] = 0.0
    u0 *= u0_max / np.abs(u0).max()
    return np.concatenate([u0.reshape(-1), np.zeros(u0.size)])


def trajectory(cfg, solver, index, wave=None):
    """One simulated trajectory (time steps, x, y, channels).

    wave: wave2d solver of one step, for plate initial conditions in WAVE_ICS.
    """
    n = cfg.train.num_variations
    if cfg.solver.name == "wave2d":
        _, _, pressure, velocity_x, velocity_y = solver.solve(
            wave_initial_condition(solver, cfg.train.ic, index, n)
        )
        state = [pressure, velocity_x, velocity_y]
    else:
        if cfg.train.ic in WAVE_ICS:
            w0 = plate_wave_initial_condition(wave, cfg.train.ic, index, n)
        else:
            w0 = initial_condition(solver, cfg.train.ic, index)
        u, v, _ = solver.solve(w0)
        state = [u, v]
    return np.stack(state, axis=-1).transpose(2, 0, 1, 3)


def simulate(cfg):
    """Simulate cfg.train.num_variations trajectories of cfg.train.num_steps_train
    steps, seeding numpy with cfg.data.seed.

    Returns a float32 array (trajectories, time steps, x, y, channels).
    """
    np.random.seed(cfg.data.seed)
    solver = make_solver(cfg)
    wave = None
    if cfg.solver.name != "wave2d" and cfg.train.ic in WAVE_ICS:
        wave = make_wave_solver(cfg, num_steps=1)
    return np.stack(
        [trajectory(cfg, solver, i, wave) for i in range(cfg.train.num_variations)]
    ).astype(np.float32)


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
