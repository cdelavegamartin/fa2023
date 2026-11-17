#!/bin/bash
# Generate the datasets of slurm/fullsweep.sbatch (CPU only, about 8 GB), for each of
# the 6 configurations of slurm/fullsweep_datasets.txt:
#   - training/validation: 1024 trajectories of 40 steps, data seed 0
#   - extrapolation (evaluate_extrapolation.py): 30 trajectories of 2000 steps, data
#     seed 5 (data.extrapolation in conf/config.yaml)
# Datasets go to $FA2023_OUTPUT/datasets (output/datasets if FA2023_OUTPUT is unset).
# Usage, from the repository root: bash slurm/generate_datasets.sh

set -euo pipefail
set -f # no globbing in the overrides

JOBS_DIR="${FA2023_OUTPUT:-output}/generate_dataset"
EXTRAPOLATION="train.num_variations=30 train.num_steps_train=2000 data.seed=5"

while read -r DATASET; do
    # shellcheck disable=SC2086
    uv run python generate_dataset.py -m $DATASET "hydra.sweep.dir=$JOBS_DIR" < /dev/null
    # shellcheck disable=SC2086
    uv run python generate_dataset.py -m $DATASET $EXTRAPOLATION "hydra.sweep.dir=$JOBS_DIR" < /dev/null
done < slurm/fullsweep_datasets.txt
