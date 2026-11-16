import torch

from platefno.nn.fno_gru import FNO_GRU_2d
from platefno.nn.fno_ref import FNO_Markov_2d
from platefno.nn.fno_rnn import FNO_RNN_2d


def build_models(cfg, size_x, size_y, device):
    """Instantiate the FGRU, FRNN and REF models of the configuration.

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
    models = (
        FNO_GRU_2d(**common),
        FNO_RNN_2d(**common, depth=cfg.nnarch.depth),
        FNO_Markov_2d(**common, depth=cfg.nnarch.depth),
    )
    return tuple(torch.nn.DataParallel(m).to(device) for m in models)
