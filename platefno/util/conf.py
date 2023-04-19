import os
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def print_hydra_config(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    return


def get_config(dir_name):
    return OmegaConf.load(os.path.join(dir_name, ".hydra", "config.yaml"))
