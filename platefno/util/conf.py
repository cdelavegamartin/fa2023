import os
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def print_hydra_config(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    return


def get_config(dir_name, config_name="config"):
    return OmegaConf.load(os.path.join(dir_name, ".hydra", f"{config_name}.yaml"))


# function to convert from the T60 format in the config to the one required by the solver
def get_t60(cfg):
    # Set T60 depending on the damping configuration
    if cfg.solver.damping.name == "nodamping":
        t60 = None
    elif cfg.solver.damping.name == "constant":
        t60 = cfg.solver.damping.T60
    elif cfg.solver.damping.name == "freqdependent":
        t60 = (
            {"f": cfg.solver.damping.f1, "T60": cfg.solver.damping.T60f1},
            {"f": cfg.solver.damping.f2, "T60": cfg.solver.damping.T60f2},
        )
    else:
        raise ValueError("Invalid damping configuration")
    return t60


if __name__ == "__main__":
    print_hydra_config()
