#!/bin/bash
# Generate the datasets of slurm/fullsweep.sbatch (CPU only, about 8 GB in
# output/datasets/), for each of the 6 configurations:
#   - training/validation: 1024 trajectories of 40 steps, data seed 0
#   - extrapolation (evaluate_extrapolation.py): 30 trajectories of 2000 steps, data
#     seed 5 (data.extrapolation in conf/config.yaml)
# The 5 plate configurations of slurm/fullsweep_datasets.txt use generate_dataset.py;
# the original 2D wave equation (wave2d_ftm) uses generate_dataset.py of a DAFX22_FNO
# checkout (`uv sync --extra cpu` there first).
# Usage, from the repository root: bash slurm/generate_datasets.sh <DAFX22_FNO checkout>

set -euo pipefail
set -f # no globbing in the overrides

DAFX22_DIR=$(realpath "${1:?usage: bash slurm/generate_datasets.sh <DAFX22_FNO checkout>}")
ROOT=$(pwd)
EXTRAPOLATION="train.num_variations=30 train.num_steps_train=2000 data.seed=5"

while read -r DATASET; do
    # shellcheck disable=SC2086
    uv run python generate_dataset.py -m $DATASET hydra.sweep.dir=output/generate_dataset < /dev/null
    # shellcheck disable=SC2086
    uv run python generate_dataset.py -m $DATASET $EXTRAPOLATION hydra.sweep.dir=output/generate_dataset < /dev/null
done < slurm/fullsweep_datasets.txt

# Paths that data.name=wave2d_ftm train.ic=impulse_random gives in conf/config.yaml
FTM="$ROOT/output/datasets/wave2d_ftm/ic_impulse_random"
(
    cd "$DAFX22_DIR"
    uv run python generate_dataset.py wave2d --steps 40 --seed 0 --output "$FTM-n_1024-steps_40-seed_0.npy"
    uv run python generate_dataset.py wave2d --variations 30 --steps 2000 --seed 5 --output "$FTM-n_30-steps_2000-seed_5.npy"
)
