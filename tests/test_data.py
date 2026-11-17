import os

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from platefno.data import channel_stats, grid_size, simulate, split, standardize

CONF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf")


def config(*overrides):
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(
            "config",
            overrides=["train.num_variations=4", "train.num_steps_train=3", *overrides],
        )


def test_simulate_shape_and_determinism():
    a, b = simulate(config()), simulate(config())
    assert a.shape == (4, 3, 40, 38, 2) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert not np.array_equal(a, simulate(config("data.seed=1")))
    assert np.array_equal(a, simulate(config("train.random_seed=1")))


@pytest.mark.parametrize("n, val_split, n_val", [(1024, 0.1, 103), (10, 0.1, 1), (4, 0.5, 2)])
def test_split_keeps_ceil_fraction_for_validation(n, val_split, n_val):
    data = np.arange(n)
    training, validation = split(data, val_split)
    assert len(validation) == n_val
    assert np.array_equal(np.concatenate([training, validation]), data)


def test_standardize_per_channel():
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((5, 3, 4, 4, 2)) * [2.0, 30.0] + [1.0, -3.0]).astype(np.float32)
    mean, std = channel_stats(data)
    out = standardize(data, mean, std).numpy()
    assert np.allclose(out.mean(axis=(0, 1, 2, 3)), 0, atol=1e-5)
    assert np.allclose(out.std(axis=(0, 1, 2, 3)), 1, atol=1e-5)
    assert np.allclose(standardize(data, [0, 0], std).numpy(), data / std, atol=1e-6)


def test_wave2d_dataset():
    cfg = config("solver=wave2d", "train.ic=impulse_random")
    data = simulate(cfg)
    assert data.shape == (4, 3, 40, 38, 3)
    assert cfg.data.name == "wave2d_ftm"
    assert grid_size(cfg) == (40, 38)


def test_output_root_follows_environment(monkeypatch):
    monkeypatch.setenv("FA2023_OUTPUT", "/scratch/user/fa2023")
    assert config().data.path.startswith("/scratch/user/fa2023/datasets/")
    monkeypatch.delenv("FA2023_OUTPUT")
    assert config().data.path.startswith("output/datasets/")
