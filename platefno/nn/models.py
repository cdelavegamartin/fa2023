import torch

from platefno.nn.fno_gru import FNO_GRU_2d
from platefno.nn.fno_ref import FNO_Markov_2d
from platefno.nn.fno_rnn import FNO_RNN_2d


MODEL_NAMES = ("gru", "rnn", "ref")


def build_models(cfg, size_x, size_y, device, seed=None):
    """Instantiate the FGRU, FRNN and REF models of the configuration.

    Returns {"gru": ..., "rnn": ..., "ref": ...}.
    With a seed, each model draws its initial weights from its own seed, so they do
    not depend on the other models (e.g. on nnarch.full_spectrum for REF).
    The models are wrapped in DataParallel, as in training, so that saved
    state_dicts keep their "module." prefix.
    """
    channels = cfg.solver.num_state_variables
    common = dict(
        in_channels=channels,
        out_channels=channels,
        spatial_size_x=size_x,
        spatial_size_y=size_y,
        width=cfg.nnarch.width,
    )
    # Configurations saved before the option existed use the original layer.
    full_spectrum = cfg.nnarch.get("full_spectrum", False)
    factories = {
        "gru": lambda: FNO_GRU_2d(**common, full_spectrum=full_spectrum),
        "rnn": lambda: FNO_RNN_2d(
            **common, depth=cfg.nnarch.depth, full_spectrum=full_spectrum
        ),
        "ref": lambda: FNO_Markov_2d(**common, depth=cfg.nnarch.depth),
    }
    models = {}
    for index, name in enumerate(MODEL_NAMES):
        if seed is not None:
            torch.manual_seed(seed * len(MODEL_NAMES) + index)
        models[name] = torch.nn.DataParallel(factories[name]()).to(device)
    return models
