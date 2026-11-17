import pytest
import torch
from omegaconf import OmegaConf

from platefno.nn.models import build_models


def config(**nnarch):
    return OmegaConf.create(
        {"solver": {"num_state_variables": 2}, "nnarch": {"depth": 3, "width": 8, **nnarch}}
    )


def has_negative_block(model):
    return any(name.endswith("weights_neg") for name, _ in model.named_parameters())


@pytest.mark.parametrize(
    "nnarch, expected",
    [({}, False), ({"full_spectrum": False}, False), ({"full_spectrum": True}, True)],
)
def test_full_spectrum_option(nnarch, expected):
    gru, rnn, ref = build_models(config(**nnarch), 40, 38, "cpu").values()
    assert has_negative_block(gru) is expected
    assert has_negative_block(rnn) is expected
    assert not has_negative_block(ref)


def test_seeded_models_do_not_depend_on_full_spectrum():
    plain = build_models(config(full_spectrum=False), 40, 38, "cpu", seed=0)
    full = build_models(config(full_spectrum=True), 40, 38, "cpu", seed=0)
    for a, b in zip(plain["ref"].parameters(), full["ref"].parameters()):
        assert torch.equal(a, b)
    assert torch.equal(plain["gru"].module.z_layer.weights, full["gru"].module.z_layer.weights)
