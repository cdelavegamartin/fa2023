"""Write the run list of slurm/fullsweep.sbatch and the dataset list of
slurm/generate_datasets.sh (the overrides that select each dataset).

Runs: 6 datasets x 6 seeds x {original, full-spectrum} FourierConv2d x {joint,
separate} gradient clipping. A joint-clipping run trains FGRU, FRNN and REF in one
task; a separate-clipping run is split into one task per model, all writing to the
same run directory.

Usage: uv run python slurm/make_run_list.py
"""

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "conf" / "experiment" / "fullspectrumsweep.yaml"

PLATE = "solver/damping=freqdependent train.ic=pluck"
WAVE = "solver/damping=nodamping solver.gamma=340.0 solver.kappa=0.0"
# Overrides that select each dataset (and its data.path)
DATASETS = {
    "plate, gamma=1, kappa=0.1": f"{PLATE} solver.gamma=1.0 solver.kappa=0.1",
    "plate, gamma=1, kappa=1": f"{PLATE} solver.gamma=1.0 solver.kappa=1.0",
    "plate, gamma=100, kappa=0.1": f"{PLATE} solver.gamma=100.0 solver.kappa=0.1",
    "plate, gamma=100, kappa=1": f"{PLATE} solver.gamma=100.0 solver.kappa=1.0",
    # Plate solver with the parameters of the 2D wave equation of Parker et al.
    "plate as 2D wave": f"{WAVE} train.ic=mix",
    # FTM 2D wave equation of Parker et al. (conf/solver/wave2d.yaml)
    "FTM 2D wave": "solver=wave2d train.ic=impulse_random",
}
SEEDS = range(6)
MODELS = ("gru", "rnn", "ref")


def runs():
    for dataset in DATASETS.values():
        for seed in SEEDS:
            for full_spectrum in ("false", "true"):
                base = (
                    f"{dataset} train.random_seed={seed} "
                    f"nnarch.full_spectrum={full_spectrum}"
                )
                yield f"{base} train.clip_jointly=true"
                for model in MODELS:
                    yield f"{base} train.clip_jointly=false train.models=[{model}]"


def swept_keys():
    """Keys with several values in the experiment's sweep; every run fixes them."""
    params = yaml.safe_load(EXPERIMENT.read_text())["hydra"]["sweeper"]["params"]
    return {key for key, value in params.items() if "," in str(value)}


def check(lines):
    required = swept_keys()
    for line in lines:
        keys = {token.split("=")[0] for token in line.split()}
        missing = required - keys
        assert not missing, f"{line!r} leaves {missing} swept"


def main():
    lines = list(runs())
    check(lines)
    (HERE / "fullsweep_runs.txt").write_text("\n".join(lines) + "\n")
    (HERE / "fullsweep_datasets.txt").write_text("\n".join(DATASETS.values()) + "\n")
    print(f"{len(lines)} runs, {len(DATASETS)} datasets")


if __name__ == "__main__":
    main()
