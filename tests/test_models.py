import pytest
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
    gru, rnn, ref = build_models(config(**nnarch), 40, 38, "cpu")
    assert has_negative_block(gru) is expected
    assert has_negative_block(rnn) is expected
    assert not has_negative_block(ref)
