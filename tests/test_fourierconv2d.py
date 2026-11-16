"""FourierConv2d spectral coverage (phd-thesis#85)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from platefno.nn.fno_ref import SpectralConv2d
from platefno.nn.layers import FourierConv2d

NX, NY, C = 40, 38, 3


def spectral_on_padded_grid(layer, x):
    """SpectralConv2d on the same (2 NX, 2 NY) zero-padded grid as FourierConv2d."""
    return layer(F.pad(x, [0, NY, 0, NX]))[..., :NX, :NY]


def reference_spectral_conv(fourier):
    """SpectralConv2d holding the weights of `fourier` (zero for a missing block)."""
    spectral = SpectralConv2d(C, C, NX, NY)
    neg = fourier.weights_neg if fourier.full_spectrum else torch.zeros_like(fourier.weights)
    with torch.no_grad():
        spectral.weights1.copy_(fourier.weights)
        spectral.weights2.copy_(neg)
    return spectral


@pytest.mark.parametrize("full_spectrum", [False, True])
def test_matches_spectralconv2d(full_spectrum):
    torch.manual_seed(0)
    fourier = FourierConv2d(C, C, NX, NY, bias=False, full_spectrum=full_spectrum)
    spectral = reference_spectral_conv(fourier)
    x = torch.randn(2, C, NX, NY)
    expected = spectral_on_padded_grid(spectral, x)
    assert torch.allclose(fourier(x), expected, atol=1e-6)


def identity_layer(full_spectrum):
    layer = FourierConv2d(1, 1, NX, NY, bias=False, full_spectrum=full_spectrum)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.view_as_real(torch.ones(p.shape[:-1], dtype=torch.cfloat)))
    return layer


def energy_ratio(layer, field):
    x = torch.tensor(field, dtype=torch.float32)[None, None]
    with torch.no_grad():
        return float(layer(x).pow(2).sum() / x.pow(2).sum())


def eigenmode(m, n):
    x, y = np.meshgrid(np.linspace(0, 1, NX), np.linspace(0, 0.95, NY), indexing="ij")
    return np.sin(m * np.pi * x) * np.sin(n * np.pi * y / 0.95)


def diagonal_wave(sign):
    x, y = np.meshgrid(np.linspace(0, 1, NX), np.linspace(0, 0.95, NY), indexing="ij")
    return np.cos(20 * x + sign * 20 * y)


def test_original_layer_drops_one_diagonal_orientation():
    layer = identity_layer(full_spectrum=False)
    assert energy_ratio(layer, diagonal_wave(+1)) > 0.9
    assert energy_ratio(layer, diagonal_wave(-1)) < 0.05
    assert energy_ratio(layer, eigenmode(2, 3)) < 0.6


@pytest.mark.parametrize("field", [diagonal_wave(+1), diagonal_wave(-1), eigenmode(2, 3)])
def test_full_spectrum_keeps_both_orientations(field):
    assert energy_ratio(identity_layer(full_spectrum=True), field) > 0.99


def test_original_layer_keeps_state_dict_keys():
    keys = set(FourierConv2d(C, C, NX, NY).state_dict())
    assert keys == {"weights", "biases"}
