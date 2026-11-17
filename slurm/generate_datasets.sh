#!/bin/bash
# Generate the 6 datasets of slurm/fullsweep.sbatch, each with data seed 0 (the run
# seeds only set the models' initial weights); CPU only, about 3.5 GB in
# output/datasets/:
#   - the 5 plate datasets of slurm/fullsweep_datasets.txt, with generate_dataset.py
#   - the original 2D wave equation with 40 steps (wave2d_ftm), with
#     generate_dataset.py of a DAFX22_FNO checkout (`uv sync --extra cpu` there first)
# Usage, from the repository root: bash slurm/generate_datasets.sh <DAFX22_FNO checkout>

set -euo pipefail
set -f # no globbing in the overrides

DAFX22_DIR=$(realpath "${1:?usage: bash slurm/generate_datasets.sh <DAFX22_FNO checkout>}")
ROOT=$(pwd)

while read -r DATASET; do
    # shellcheck disable=SC2086
    uv run python generate_dataset.py -m $DATASET hydra.sweep.dir=output/generate_dataset
done < slurm/fullsweep_datasets.txt

# Path that data.name=wave2d_ftm train.ic=impulse_random gives in conf/config.yaml
OUTPUT="$ROOT/output/datasets/wave2d_ftm/ic_impulse_random-n_1024-steps_40-seed_0.npy"
(cd "$DAFX22_DIR" && uv run python generate_dataset.py wave2d --steps 40 --seed 0 --output "$OUTPUT")
